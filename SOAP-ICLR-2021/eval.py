import torch, torchvision
import torch.nn as nn
from torchvision import transforms

import json
import os
from functools import partial

from vgg import VGG
from train import pi_criterion


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
def evaluate_adversarial(model, loader, aux_criterion, purify):

    model.eval()
    error, acc = 0., 0.
    for X, y in loader:

        X_pfy = purify(model, aux_criterion, X)
        pred = model(X_pfy)

        loss = nn.functional.cross_entropy(pred, y)
        error += loss.item() 
        acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(loader)
    acc = acc / len(loader.dataset)
    print('adv loss: {} / acc: {}'.format(error, acc))

    return acc


def defense_wrapper(model, criterion, X, defense, epsilon=None, step_size=None, num_iter=None):
    
    model.aux = True
    if defense == 'fgsm':
        inv_delta = fgsm(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon)
    elif defense == 'pgd_linf':
        inv_delta = pgd_linf(model, lambda model, X: -criterion(model, X), X, epsilon=epsilon, step_size=step_size, num_iter=num_iter)
    elif defense == 'inject_noise':
        inv_delta = inject_noise(X, epsilon)
    else:
        raise TypeError("Unrecognized defense name: {}".format(defense))
    model.aux = False
    # model.eval()
    return inv_delta


def purify(model, aux_criterion, X, defense_mode='pgd_linf', delta=4/255, step_size=4/255, num_iter=5):

    if aux_criterion is None:
        return X
    aux_track = torch.zeros(11, X.shape[0])
    inv_track = torch.zeros(11, *X.shape)
    for e in range(11):
        defense = partial(defense_wrapper, criterion=aux_criterion, defense=defense_mode, epsilon=e*delta, step_size=step_size, num_iter=num_iter)
        inv_delta = defense(model, X=X)
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
pfy = partial(purify, defense_mode=args.defense, delta=args.pfy_delta, step_size=args.pfy_step_size, num_iter=args.pfy_num_iter)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))])
testset = torchvision.datasets.CIFAR10(root=config["data_path"], train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"],
                                         shuffle=False, num_workers=2)


evaluate(model, testloader, criterion)
evaluate_adversarial(model, testloader, criterion, aux_criterion, pfy)
