import os.path as osp

import mmcv
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from mmgen.models.builder import MODELS, build_module
from mmgen.models.misc import tensor2img
from ..common import set_requires_grad
from .base_gan import BaseGAN


@MODELS.register_module()
class AgileEncoder(nn.Module):
    def __init__(self, encoder, decoder, loss=None, train_cfg=None,
                 test_cfg=None,pretrained=None):
        super().__init__()
        self._encoder_cfg = deepcopy(encoder)
        self.encoder = build_module(encoder)
        self._decoder_cfg = deepcopy(decoder)
        self.decoder = build_module(decoder)
    def forward(self, x):
        code, _, _ = self.encoder(x)
        code = code.unbind(dim=1)
        rec_x = self.decoder(code, input_is_latent=True)
        return rec_x
    
    
# @MODELS.register_module()
# class AgileTransfer

# @MODELS.register_module()
# class AgileTranslation