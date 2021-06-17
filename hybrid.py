import torch
import torch.nn.functional as F


def gen_loss(clean, adv, network1, network2, network3):
    """
        clean: a batch of clean sample
        adv: a batch of adversarial sample with respect to clean
        network1:  purifier network
        network2: critic network
        network3: feature extractor
    """
    p1, p2, p3 = 5e-3, 0.01, 1

    purified = network1(adv)

    loss_adv = -torch.mean(torch.log(F.sigmoid(network2(purified) - network2(clean))))
    loss_img = F.mse_loss(clean, purified)
    loss_feat = F.l1_loss(network3(purified), network3(clean))

    loss = p1 * loss_adv + p2 * loss_img + p3 * loss_feat

    return loss


def dis_loss(clean, adv, network1, network2):
    """
        clean: a batch of clean sample
        adv: a batch of adversarial sample with respect to clean
        network1:  purifier network
        network2: critic network
    """
    purified = network1(adv)
    loss_des = -torch.mean(torch.log(F.sigmoid(network2(clean) - network2(purified))))

    return loss_des
