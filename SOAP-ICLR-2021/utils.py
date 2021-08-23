import torch
import torch.nn as nn


def random_trans_combo(tensor, df=False):
    # tensor: bs * c * h * w
    if not df:
        tensor += (torch.randn_like(tensor)*0.1).clamp(0,1)
    if torch.rand(1) > 0.5 or df:
        tensor = tensor.flip(3)
    if not df:
        r_h = torch.randint(0, 8, (1,)).item()
        r_w = torch.randint(0, 8, (1,)).item()
        h = torch.randint(24, 32-r_h, (1,))
        w = torch.randint(24, 32-r_w, (1,))
    else:
        r_h, r_w, h, w = 2, 2, 28, 28
    tensor = tensor[:, :, r_h:r_h+h, r_w:r_w+w]
    return nn.functional.interpolate(tensor, [32, 32])


def pi_criterion(model, X, joint=False, train=False, reduction='mean'):
    if not train:
        X1 = X
    else:
        X1 = random_trans_combo(X)
    X2 = random_trans_combo(X, df=~joint)
    l1, l2 = model(X1), model(X2)
    l = torch.cat((l1, l2), dim=0)
    loss = nn.functional.mse_loss(l1, l2, reduction=reduction)
    if not joint:
        return loss
    return loss, l
