import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json

from model import Score


config = None
with open('config.json', 'r') as f:
    config = json.load(f)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))])

batch_size = config["batch_size"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trainset = torchvision.datasets.CIFAR10(root=config["data_path"], train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=config["data_path"], train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def score_loss(model, samples, sigma=0.1):

    a = torch.randn_like(samples) * sigma
    perturbed_samples = samples + a
    scores = model(perturbed_samples)
    scores = scores.view(scores.shape[0], -1) * sigma ** 2
    scores += a.view(a.shape[0], -1)

    target = samples.view(samples.shape[0], -1)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss

net = Score().to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)

for epoch in range(config["epoch"]):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss = score_loss(net, inputs)

        loss.backward()
        optimizer.step()
        # print statistics
        # print(loss.item())
        running_loss += loss.item()
        if i % 500 == 499:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.10f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')

torch.save(net, config["save_path"])
