"""
Self supervised attack is purely based on maximizing the perceptual feature difference.
"""
import torch
import torch.nn.functional as F


def get_adv(clean, extractor, epsilon=0.06):
    # generate adversarial example
    x_adv = clean.detach() + 0.001 * torch.randn(clean.shape).cuda().detach()
    # distance == 'l_inf'
    x_adv.requires_grad_()
    with torch.enable_grad():
        y_adv = extractor(x_adv)
        y_clean = extractor(clean)
        loss = F.mse_loss(y_adv, y_clean)
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x_adv.detach() + epsilon * torch.sign(grad.detach())
    x_adv = torch.min(torch.max(x_adv, clean - epsilon), clean + epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv
