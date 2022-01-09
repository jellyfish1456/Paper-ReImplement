import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()


def atk(x, y, model, eps):
    x = x.clone().detach()
    y = y.clone().detach()
    x.requires_grad = True
    out = model(x)
    cost = loss(out, y)
    grad = torch.autograd.grad(cost, x, retain_graph=False,
                               create_graph=False)[0]
    adv_images = x + eps / 255 * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images, out


def run(net, train_loader, optimizer, scheduler, device, cfg):
    net.train()
    epoch = cfg["epoch"]

    for i in range(epoch):
        for j, data in enumerate(train_loader):
            imgs, label = data
            imgs = imgs.to(device)
            label = label.to(device)
            bs = len(imgs)

            x_adv, logit_clean = atk(imgs, label, net, cfg["eps"])
            pert = (x_adv - imgs).detach()

            _, pre_clean = torch.max(logit_clean.data, 1)
            correct = (pre_clean == label)

            correct_idx = torch.masked_select(
                torch.arange(bs).to(device), correct)
            wrong_idx = torch.masked_select(torch.arange(bs).to(device), ~correct)

            x_adv[wrong_idx] = imgs[wrong_idx]

            c = cfg["c"]
            inf_batch = cfg["inf_batch"]

            Xs = (torch.cat([imgs] * (c - 1)) +
                  torch.cat([torch.arange(1, c).to(device).view(-1, 1)] * bs,
                            dim=1).view(-1, 1, 1, 1) * torch.cat([pert / c] *
                                                                 (c - 1)))
            Ys = torch.cat([label] * (c - 1))

            idx = correct_idx
            idxs = []
            net.eval()

            with torch.no_grad():
                for k in range(c - 1):
                    if len(idx) == 0:
                        break
                    elif inf_batch >= len(idx) * (len(idxs) + 1):
                        idxs.append(idx + k * bs)
                    else:
                        pass

                    if inf_batch < len(idx) * (len(idxs) + 1) or k == c - 2:
                        idxs = torch.cat(idxs).to(device)
                        pre = net(Xs[idxs]).detach()
                        _, pre = torch.max(pre.data, 1)
                        correct = (pre == Ys[idxs]).view(-1, len(idx))
                        max_idx = idxs.max() + 1
                        wrong_idxs = (idxs.view(-1, len(idx)) *
                                      (1 - correct * 1)) + (max_idx *
                                                            (correct * 1))

                        wrong_idx, _ = wrong_idxs.min(dim=0)
                        wrong_idx = torch.masked_select(wrong_idx,
                                                        wrong_idx < max_idx)
                        update_idx = wrong_idx % bs
                        x_adv[update_idx] = Xs[wrong_idx]

                        # Set new indexes by eliminating updated indexes.
                        idx = torch.tensor(list(set(idx.cpu().data.numpy().tolist())\
                                            -set(update_idx.cpu().data.numpy().tolist())))
                        idxs = []

            net.train()
            pre = net(x_adv.detach())
            cost = loss(pre, label)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if j % 300 == 99:
                print("epoch : {}, loss is ".format(i), cost.item())

        scheduler.step()
