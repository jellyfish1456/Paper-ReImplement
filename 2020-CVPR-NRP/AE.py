import torch
import torch.nn as nn
# from torchvision import transforms


class Purifier(nn.Module):
    def __init__(self):
        super(Purifier, self).__init__()

        self.mean = torch.tensor(
            [[0.4913997551666284, 0.48215855929893703, 0.4465309133731618]])
        self.std = torch.tensor(
            [[0.24703225141799082, 0.24348516474564, 0.26158783926049628]])

        # self.transform = transforms.Compose(
        #     [transforms.Normalize(mean=self.mean, std=self.std)])
        self.pad = (1, 1)

        self.conv_12 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn_12 = nn.BatchNorm2d(16)

        self.conv_23 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn_23 = nn.BatchNorm2d(32)

        self.conv_34 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_34 = nn.BatchNorm2d(64)

        self.conv_43 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn_43 = nn.BatchNorm2d(32)

        self.conv_32 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn_32 = nn.BatchNorm2d(16)

        self.conv_21 = nn.Conv2d(16, 3, 3, padding=1)
        self.bn_21 = nn.BatchNorm2d(3)

        self.pool = nn.MaxPool2d(3, padding=1, stride=1)

    def forward(self, x):
        # print(x.size())
        x = self.conv_12(x)
        x = self.bn_12(self.pool(x))
        x = self.bn_23(self.pool(self.conv_23(x)))
        x = self.bn_34(self.pool(self.conv_34(x)))

        x = self.bn_43(self.pool(self.conv_43(x)))
        x = self.bn_32(self.pool(self.conv_32(x)))
        x = self.bn_21(self.pool(self.conv_21(x)))

        return x


if __name__ == "__main__":
    # batch, channel, height, width
    data = torch.rand(16, 3, 32, 32)
    model = Purifier()
    y = model(data)