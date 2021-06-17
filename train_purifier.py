"""
purifier and critic training script
"""
import os
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms

from hybrid import gen_loss, dis_loss
from AE import Purifier
from critic import Discriminator
from feature import get_extractor
from ssp import get_adv

# setup GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 64
test_batch_size = 64

trainset = torchvision.datasets.CIFAR100(
    root='cifar100_ogd/datasets',
    train=True,
    download=False,
    transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=batch_size,
                                           shuffle=True)
testset = torchvision.datasets.CIFAR100(
    root='cifar100_ogd/datasets',
    train=False,
    download=False,
    transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=test_batch_size,
                                          shuffle=False)

# prepare purifier,critic and feature extractor
# channel first (batch,channel,width,height)
purifier = Purifier().to(device)
critic = Discriminator().to(device)
extractor_path = "cifar100_data/checkpoint/resnet34_ckpt.pth"
extractor = get_extractor(extractor_path).to(device)


def train_purifier(optimizer, epochs):
    """
    optimizer : optimize purifier
    epochs : epochs for training purifier in a complete training process
    """
    purifier.train()
    for epoch in range(0, epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            # print(data.size())
            # data = data.transpose(1, 3)
            clean = data.to(device)
            # print(clean.size())
            # clean = clean
            adv = get_adv(clean, extractor).to(device)
            optimizer.zero_grad()
            loss = gen_loss(clean, adv, purifier, critic, extractor)
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % 100 == 0:
                print("-------Purifier training process-------")
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def train_critic(optimizer, epochs):
    """
    optimizer : optimize discriminator
    epochs : epochs for training discriminator in a complete training process
    """
    critic.train()
    for epoch in range(0, epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            # print(data.size())
            # data = data.transpose(1, 3)
            clean = data.to(device)
            # print(clean.size())
            # clean
            adv = get_adv(clean, extractor)
            optimizer.zero_grad()
            loss = dis_loss(clean, adv, purifier, critic)
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % 100 == 0:
                print("-------Discriminator training process--------")
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def main():
    gen_optimizer = optim.Adam(purifier.parameters(),
                               lr=0.01,
                               weight_decay=3.5e-3)
    dis_optimizer = optim.SGD(critic.parameters(),
                              lr=0.01,
                              weight_decay=3.5e-3)
    epochs = 100
    for epoch in range(1, epochs):
        # one complete training process
        train_critic(dis_optimizer, 1)
        train_purifier(gen_optimizer, 1)
    torch.save(purifier, 'purifier.pkl')


if __name__ == '__main__':

    main()
