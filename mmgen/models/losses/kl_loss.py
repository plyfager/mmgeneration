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
        batch = logvar.shape[0]
        logvar = logvar.view(batch, -1)
        mu = mu.view(batch, -1)
        loss = torch.mean(0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return loss * self.loss_weight
