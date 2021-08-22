import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import json
import os
from functools import partial

from vgg import VGG


def joint_criterion(model, aux_criterion, X, y, alpha=1.):
    aux_loss, l = aux_criterion(model, X, joint=True, train=True)

    y = y.repeat(2)

    loss = nn.functional.cross_entropy(l, y)
    
    return loss + aux_loss * alpha, (loss, aux_loss)


def random_trans_combo(tensor, df=False):
    # tensor: bs * c * h * w
    if not df:
        tensor += (torch.randn_like(tensor)*0.1).clamp(0,1)
    if torch.rand(1) > 0.5 or df:
        tensor = tensor.flip(3)
    if not df:
        r_h = torch.randint(0, 8, (1,)).item()
        r_w = torch.randint(0, 8, (1,)).item()
        h = torch.randint(24, 32-r_h, (1,))
        w = torch.randint(24, 32-r_w, (1,))
    else:
        r_h, r_w, h, w = 2, 2, 28, 28
    tensor = tensor[:, :, r_h:r_h+h, r_w:r_w+w]
    return nn.functional.interpolate(tensor, [32, 32])


def pi_criterion(model, X, joint=False, train=False, reduction='mean'):
    if not train:
        X1 = X
    else:
        X1 = random_trans_combo(X)
    X2 = random_trans_combo(X, df=~joint)
    l1, l2 = model(X1), model(X2)
    l = torch.cat((l1, l2), dim=0)
    loss = nn.functional.mse_loss(l1, l2, reduction=reduction)
    if not joint:
        return loss
    return loss, l


def train_with_auxiliary(model, train_loader, joint_criterion, optimizer, device):
    
    model.train()
    error, acc, error_aux, acc_aux = 0., 0., 0., 0.
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        joint_loss, (loss, aux_loss) = joint_criterion(model, X=X, y=y)
        error += loss.item()
        error_aux += aux_loss.item()
        
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()
            
        acc += (model.pred[:y.shape[0]].max(dim=1)[1] == y).sum().item()

    error = error / len(train_loader)
    error_aux = error_aux / len(train_loader)
    acc = acc / len(train_loader.dataset)
    if joint_criterion.keywords['aux_criterion'].__name__ == 'rotate_criterion':
        acc_aux = acc_aux / len(train_loader.dataset) / 4
        print('train loss: {} / acc: {} / err-aux: {} / acc-aux: {}'.format(error, acc, error_aux, acc_aux))
    else:
        print('train loss: {} / acc: {} / err-aux: {}'.format(error, acc, error_aux))


config = None
with open('config.json', 'r') as f:
    config = json.load(f)

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")


model = VGG('VGG16')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_size = config["batch_size"]
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.1)
criterion = partial(joint_criterion, aux_criterion=pi_criterion, alpha=1)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))])
trainset = torchvision.datasets.CIFAR10(root=config["data_path"], train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

print('Start training...')
for epoch in range(100):
    print('epoch: {}'.format(epoch))
    train_with_auxiliary(model, trainloader, criterion, optimizer, scheduler, device)
    scheduler.step()

torch.save(model.state_dict(), config["save_path"])
