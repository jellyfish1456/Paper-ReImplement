import torch
from resnet18 import ResNet18
from collections import OrderedDict
from torch.utils.data import Dataset
import numpy as np
from vgg import VGG

mean = 0.4738999871706486


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
            x += mean

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


print('load model')
net = VGG('VGG16')
pthfile = r'VGG16_ckpt.pth'
d = torch.load(pthfile, map_location=torch.device('cpu'))['net']
d = OrderedDict([(k[7:], v) for (k, v) in d.items()])
net.load_state_dict(d)
# 加载 purifier
print('load purifier')
purfier = torch.load('NRP/purifier.pkl',
                     map_location='cpu')
purfier.eval()
print('load data')
dataPath = "data/cifar10_data.npy"
labelPath = "data/cifar10_label.npy"

dataset = TensorDataset(dataPath, labelPath)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=0)

total = 0
correct = 0
print('natural acc: ', end=', ')
for batch_idx, (inputs, targets) in enumerate(dataloader):
    # inputs = purfier(inputs)
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
acc = 100. * correct / total
print(acc)

total = 0
correct = 0
print('natural acc: ', end=', ')
for batch_idx, (inputs, targets) in enumerate(dataloader):
    inputs = purfier(inputs)
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
acc = 100. * correct / total
print(acc)

data_root_path = 'adv_data/cifar10/'
adv_file = ['fgsm/', 'bim/', 'cw/']
perturbation = ['0.015x_adv.npy', '0.03x_adv.npy', '0.06x_adv.npy']

data_load = ""

# 开始对抗训练
for adv in adv_file:
    for path_ in perturbation:
        adv_data_path = data_root_path + adv + path_
        # print(adv, '  ', path_[0:-8], end=', ')

        advdata = TensorDataset(adv_data_path, labelPath, is_adv=True)
        adv_loader = torch.utils.data.DataLoader(advdata, batch_size=128)
        print('to pridict: ', adv_data_path, end=', ')
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(adv_loader):
            inputs = purfier(inputs)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print(acc)
