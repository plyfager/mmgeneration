import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODULES


@MODULES.register_module()
class KLloss(nn.Module):

    def __init__(self, loss_weight=5e-4):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, logvar, mu):
        dim = logvar.shape[-1]
        logvar = logvar.view(-1, dim)
        mu = mu.view(-1, dim)
        loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        return loss * self.loss_weight
