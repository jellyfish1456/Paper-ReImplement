import torch, os, json
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import model, train

if __name__ == "__main__":

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    with open("config.json", "r") as f:
        cfg = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPU"]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    data_path = cfg["data_path"]
    batch_size = cfg["batch_size"]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root=data_path,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path,
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)

    net = model.get_model("sample", 10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[60, 120, 160],
                                               gamma=0.1)
    print("ready to train")
    train.run(net, trainloader, optimizer, scheduler, device, cfg)

    model_path = cfg["save_path"]
    torch.save(net.state_dict(), model_path)
