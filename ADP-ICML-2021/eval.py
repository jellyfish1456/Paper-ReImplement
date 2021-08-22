import torch, torchvision
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from model import Score
import json
from vgg import VGG


mean = 0.4738999871706486
config = None
with open('config.json', 'r') as f:
    config = json.load(f)

print('============, load data')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))])

batch_size = config["batch_size"]

trainset = torchvision.datasets.CIFAR10(root=config["data_path"], train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=config["data_path"], train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

print('============, load purify model')
purfier = Score()
purfier.load_state_dict(torch.load(config['save_path'], map_location=torch.device('cpu')))
purfier.eval()


print('============, load target model')
net = VGG('VGG16')
pthfile = r'VGG16_ckpt.pth'
d = torch.load(pthfile, map_location=torch.device('cpu'))['net']
d = OrderedDict([(k[7:], v) for (k, v) in d.items()])
net.load_state_dict(d)

total = 0
correct = 0
print('============, natural acc: ', end=' ')
for batch_idx, (inputs, targets) in enumerate(testloader):
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
acc = 100. * correct / total
print(acc)


def score_test(testloader):

    delta = 1e-5
    lambda_ = 0.05
    tao = 0.001
    alpha = None
    correct = 0
    total = 0
    sigma = 0.1

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):

            inputs, labels = data
            predicted = torch.zeros(10, inputs.size()[0])

            for i in range (10):

                inputs += torch.randn_like(inputs) * sigma
                torch.clamp(inputs, 0.0, 1.0)

                for _ in range (10):

                    denoise = purfier(inputs)

                    inputs_ = inputs + delta * denoise

                    f1 = torch.bmm(inputs.view(inputs.size()[0], 1, -1), inputs_.view(inputs_.size()[0], -1, 1))
                    f2 = torch.bmm(inputs.view(inputs.size()[0], 1, -1), inputs.view(inputs_.size()[0], -1, 1))
                    f = torch.div(f1, f2)

                    alpha = torch.squeeze(torch.clamp(lambda_ * delta / (1.0 - f), min=0.00001, max=0.1))
                    inputs = torch.clamp(inputs.detach() + alpha * denoise, 0.0, 1.0)

                    if (torch.norm(purfier(inputs), p=1) < tao):
                        break

                outputs = net(inputs)
                _, idx = torch.max(outputs.data, 1)
                predicted[i] = idx

            predicted, _ = torch.mode(predicted, 0)
            # break
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # break
        
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

# score_test(testloader)


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


data_root_path = '/home/liyanni/1307/zwh/defense/adv_data/cifar10/'
adv_file = ['fgsm/', 'bim/', 'cw/']
perturbation = ['0.015x_adv.npy', '0.03x_adv.npy', '0.06x_adv.npy']
labelPath = "/home/liyanni/1307/zwh/whiteBox/data/cifar10_label.npy"

data_load = ""

# 开始对抗去噪
for adv in adv_file:
    for path_ in perturbation:
        adv_data_path = data_root_path + adv + path_
        # print(adv, '  ', path_[0:-8], end=', ')

        advdata = TensorDataset(adv_data_path, labelPath, is_adv=True)
        adv_loader = torch.utils.data.DataLoader(advdata, batch_size=32)
        print('to pridict: ', adv_data_path, end=', ')
        score_test(adv_loader)
