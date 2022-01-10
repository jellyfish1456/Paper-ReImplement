import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")


com = [
    16,
    "ELU",
    32,
    "ELU",
    64,
    "ELU",
    128,
    "ELU",
    256,
    "ELU",
    128,
    "ELU",
    64,
    "ELU",
    32,
    "ELU",
    4,
]
rec = [
    "SIGMOID",
    32,
    "ELU",
    64,
    "ELU",
    128,
    "ELU",
    256,
    "ELU",
    128,
    "ELU",
    64,
    "ELU",
    32,
    "ELU",
    16,
    "ELU",
    3,
]


class ComDefend(nn.Module):
    def __init__(self, phi=20):
        super(ComDefend, self).__init__()
        self.name = "ComDefend"
        self.phi = phi
        self.com, self.rec = self._make_layers(com, rec)

    def _make_layers(self, com, rec):
        comCNN = []
        recCNN = []
        inchannels = 3

        for x in com:
            if x == "ELU":
                comCNN.append(nn.ELU())
            elif x == "SIGMOID":
                comCNN.append(nn.Sigmoid())
            else:
                comCNN.append(nn.Conv2d(inchannels, x, kernel_size=3, padding=1))
                comCNN.append(nn.BatchNorm2d(x))
                inchannels = x

        for x in rec:
            if x == "ELU":
                recCNN.append(nn.ELU())
            elif x == "SIGMOID":
                recCNN.append(nn.Sigmoid())
            else:
                recCNN.append(nn.Conv2d(inchannels, x, kernel_size=3, padding=1))
                recCNN.append(nn.BatchNorm2d(x))
                inchannels = x

        return nn.Sequential(*comCNN), nn.Sequential(*recCNN)

    def forward(self, x):
        hidden = self.com(x)
        hidden_gauss = hidden + (self.phi ** 0.5) * torch.randn(hidden.shape).to(device)
        output = self.rec(hidden_gauss)

        return hidden, output


comDefend = ComDefend().to(device)


import numpy as np

data = (
    np.load("/home/liyanni/1307/zwh/whiteBox/data/cifar10_data.npy").astype("float32")
    / 255
)
data = data.transpose((0, 3, 1, 2))
label = np.load("/home/liyanni/1307/zwh/whiteBox/data/cifar10_label.npy")
label = np.reshape(label, (label.shape[0],))


def dataset(batch=50):
    train = data[:50000]
    for i in range(len(train) // batch):
        yield train[i * batch : (i + 1) * batch], train[i * batch : (i + 1) * batch]


comDefend.train()
from torch.autograd import Variable

criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, comDefend.parameters()))
for epoch in range(20):
    dataGen = dataset()
    l1 = 0.0
    l2 = 0.0
    for i, (x, y) in enumerate(dataGen, 1):
        # get the inputs
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x, y = x.to(device), y.to(device)
        x, y = Variable(x, requires_grad=True), Variable(y)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        o1, o2 = comDefend(x)
        # loss
        loss1 = criterion1(o1, torch.zeros(o1.shape).to(device))
        loss2 = criterion2(o2, y)
        loss = 0.0001 * loss1 + loss2
        # backward
        loss.backward()
        # update weights
        optimizer.step()
        # print statistics
        l1 += loss1.item()
        l2 += loss2.item()

        if i % 50 == 0:  # print every 2000 mini-batches
            print("epoch:{}  loss1:{}  loss2{}".format(epoch, l1 / 50.0, l2 / 50.0))
            l1, l2 = 0.0, 0.0


from models.resnet18 import ResNet50
from collections import OrderedDict

target = ResNet50()
pthfile = r"checkpoints/resnet50_ckpt.pth"
d = torch.load(pthfile)["net"]
d = OrderedDict([(k[7:], v) for (k, v) in d.items()])
target.load_state_dict(d)
target.to(device)

mean = np.mean(data, axis=0)

adv_x = (
    np.load("../adv_data/cifar10/cw/0.015x_adv.npy")
    .astype("float32")
    .transpose(0, 3, 1, 2)
)
adv_x = adv_x + mean


def testGen(data, batch=50):
    testx = data[50000:]
    testy = label[50000:]
    for i in range(len(testx) // batch):
        yield testx[i * batch : (i + 1) * batch], testy[i * batch : (i + 1) * batch]


test = testGen(adv_x)
correct = 0
total = 0
for inputs, labels in test:
    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
    inputs, labels = inputs.to(device), labels.to(device)
    _, outputs = comDefend(inputs)
    outputs = target(outputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("Accuracy {}".format(100.0 * correct / total))
