import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset

import json
import numpy as np

import model

# 干净样本准确率
def evaluate(model, eval_loader):

    error, acc = 0.0, 0.0
    with torch.no_grad():
        for X, y in eval_loader:

            pred = model(X)
            acc += (pred.max(dim=1)[1] == y).sum().item()

    acc = acc / len(eval_loader.dataset)
    print("acc: {}".format(acc))

    return acc


config = None
with open("config.json", "r") as f:
    config = json.load(f)

model = model.get_model("sample", 10)
model.load_state_dict(torch.load(config["save_path"], map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

testset = torchvision.datasets.CIFAR10(
    root=config["data_path"], train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config["batch_size"], shuffle=False, num_workers=2
)

print("clean acc", end=", ")
evaluate(model, testloader)

data_root_path = "/home/liyanni/1307/zwh/defense/adv_data/cifar10/"
adv_file = ["fgsm/", "bim/", "cw/"]
perturbation = ["0.015x_adv.npy", "0.03x_adv.npy", "0.06x_adv.npy"]
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
            # pass
            x += 0.47

        x = x.astype("float32")
        data = x.transpose(0, 3, 1, 2)
        label = np.load(labelPath)[50000:]
        label = np.reshape(label, (data.shape[0],))
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
        print("to pridict: ", adv_data_path, end=", ")
        evaluate(model, adv_loader)
