import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import json
import os
from functools import partial

from vgg import VGG
from utils import pi_criterion


def joint_criterion(model, aux_criterion, X, y, alpha=1.):
    aux_loss, l = aux_criterion(model, X, joint=True, train=True)

    y = y.repeat(2)

    loss = nn.functional.cross_entropy(l, y)

    return loss + aux_loss * alpha, (loss, aux_loss)


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

    error = error / len(train_loader)
    error_aux = error_aux / len(train_loader)

    print('train loss: {} / err-aux: {}'.format(error, error_aux))


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

for epoch in range(config["epochs"]):
    print('epoch: {}'.format(epoch), end=' ')
    train_with_auxiliary(model, trainloader, criterion, optimizer, device)
    scheduler.step()

torch.save(model.state_dict(), config["save_path"])
