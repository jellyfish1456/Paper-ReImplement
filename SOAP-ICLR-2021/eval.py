import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

import json
import numpy as np
from functools import partial

from vgg import VGG
from utils import pi_criterion
from cw_attack import L2Adversary


# 干净样本准确率
def evaluate(model, eval_loader, criterion):

    error, acc = 0., 0.
    with torch.no_grad():
        for X, y in eval_loader:

            pred = model(X)

            loss = criterion(pred, y)
            error += loss.item()

            acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(eval_loader)
    acc = acc / len(eval_loader.dataset)
    print('val loss: {} / acc: {}'.format(error, acc))

    return acc


# 对抗样本准确率
def evaluate_adversarial(model, loader, aux_criterion, purify, mode):

    model.eval()
    error, acc = 0., 0.
    for X, y in loader:

        X_pfy = purify(model, aux_criterion, X, mode)
        pred = model(X_pfy)

        loss = nn.functional.cross_entropy(pred, y)
        error += loss.item() 
        acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(loader)
    acc = acc / len(loader.dataset)
    print('adv loss: {} / acc: {}'.format(error, acc))

    return acc


def fgsm(model, criterion, X, y=None, epsilon=0.1, bound=(0,1)):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)

    loss = criterion(model, X + delta)

    loss.backward()

    delta = epsilon * delta.grad.detach().sign()

    return (X + delta).clamp(*bound) - X


def cw(model, X, y=None, epsilon=0.1, num_classes=10):
    delta = L2Adversary()(model, X.clone().detach(), y, num_classes=num_classes).to(X.device) - X
    delta_norm = torch.norm(delta, p=2, dim=(1,2,3), keepdim=True) + 1e-4
    delta_proj = (delta_norm > epsilon) * delta / delta_norm * epsilon + (delta_norm < epsilon) * delta
    return delta_proj


def bim(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40):
    """ Construct PGD adversarial examples on the examples X"""

    delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):

        loss = criterion(model, X + delta)

        loss.backward()
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        delta.grad.zero_()

    delta.data = (X + delta).clamp(*bound) - X
    return delta.detach()


def defense_wrapper(model, criterion, X, mode, epsilon=None, step_size=None, num_iter=None):
    inv_delta = None
    if mode == 'fgsm':
        inv_delta = fgsm(model, lambda model, X: -criterion(model, X), X)
    elif mode == 'cw':
        inv_delta = cw(model=model, X=X)
    else:
        inv_delta = bim(model, lambda model, X: -criterion(model, X), X)
    # model.eval()
    return inv_delta


def purify(model, aux_criterion, X, mode):

    if aux_criterion is None:
        return X
    aux_track = torch.zeros(11, X.shape[0])
    inv_track = torch.zeros(11, *X.shape)
    for e in range(11):
        defense = partial(defense_wrapper, criterion=aux_criterion, mode=mode, 
            epsilon=e*4/255, step_size=4/255, num_iter=5)
        inv_delta = defense(X=X)
        inv_track[e] = inv_delta
        aux_track[e, :] = aux_criterion(model, (X+inv_delta).clamp(0,1)).detach()
    e_selected = aux_track.argmin(dim=0)
    return inv_track[e_selected, torch.arange(X.shape[0])].to(X.device) + X


config = None
with open('config.json', 'r') as f:
    config = json.load(f)

model = VGG('VGG16')
model.load_state_dict(torch.load(config['save_path'], map_location=torch.device('cpu')))
model.eval()

aux_criterion = pi_criterion
criterion = nn.CrossEntropyLoss()
pfy = partial(purify)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))])
testset = torchvision.datasets.CIFAR10(root=config["data_path"], train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"],
                                         shuffle=False, num_workers=2)


evaluate(model, testloader, criterion)


data_root_path = '/home/liyanni/1307/zwh/defense/adv_data/cifar10/'
adv_file = ['fgsm/', 'bim/', 'cw/']
perturbation = ['0.015x_adv.npy', '0.03x_adv.npy', '0.06x_adv.npy']
labelPath = "/home/liyanni/1307/zwh/whiteBox/data/cifar10_label.npy"


class TensorDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """
    def __init__(self, dataPath, labelPath, is_adv=False):
        x = np.load(dataPath)
        x = x[50000:]
        if is_adv:
            pass
            # x += 0.47

        x = x.astype("float32")
        data = x.transpose(0, 3, 1, 2)
        label = np.load(labelPath)[50000:]
        label = np.reshape(label, (data.shape[0], ))
        data, label = torch.from_numpy(data), torch.from_numpy(label)
        self.data_tensor = data
        self.target_tensor = label

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


# 开始对抗去噪
for adv in adv_file:
    for path_ in perturbation:
        adv_data_path = data_root_path + adv + path_

        advdata = TensorDataset(adv_data_path, labelPath, is_adv=True)
        adv_loader = torch.utils.data.DataLoader(advdata, batch_size=128)
        print('to pridict: ', adv_data_path, end=', ')
        evaluate_adversarial(model, testloader, aux_criterion, pfy, mode=adv_file[:-1])
