import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json, os

from model import Score


print('==========, load model config')
config = None
with open('config.json', 'r') as f:
    config = json.load(f)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))])

batch_size = config["batch_size"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('==========, load data config')
trainset = torchvision.datasets.CIFAR10(root=config["data_path"], train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=config["data_path"], train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

sigmas = torch.tensor(
    np.exp(np.linspace(np.log(1), np.log(0.01), 10))).float().to(device)


def score_loss(model, samples, labels, sigmas):
    sigma = sigmas[labels].view(samples.shape[0])

    a = torch.einsum("iabc,i->iabc", [torch.randn_like(samples), sigma])

    scores = model(samples + a)

    scores = torch.einsum("iabc,i->iabc", [scores, sigma ** 2]).view(scores.shape[0], -1) + a.view(a.shape[0], -1)
    target = samples.view(samples.shape[0], -1)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss


print('==========, create model')
net = Score().to(device)
if os.path.exists(config["save_path"]) and config["continue_train"]:
    net.load_state_dict(torch.load(config['save_path']))

optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)


min_loss = 1000000000
print('==========, begin to train')
for epoch in range(config["epoch"]):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        labels = torch.randint(0, len(sigmas), (inputs.shape[0],)).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        loss = score_loss(net, inputs, labels, sigmas)

        loss.backward()
        optimizer.step()
        # print statistics
        # print(loss.item())
        running_loss += loss.item()
        if i % 100 == 99:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.10f' %
                  (epoch + 1, i + 1, running_loss / 100))
            if (running_loss / 100 < min_loss):
                min_loss = running_loss / 100
                torch.save(net.state_dict(), config["save_path"])
            running_loss = 0.0

print('Finished Training and save model')
