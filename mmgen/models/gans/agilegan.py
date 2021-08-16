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
        
        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()
            
    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part
        
    def forward(self, x):
        code, _, _ = self.encoder(x)
        code = code.unbind(dim=1)
        rec_x = self.decoder(code, input_is_latent=True)
        return rec_x
    
    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None):
        # get data from data_batch
        real_imgs = data_batch[self.real_img_key]
        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # encoder training
        set_requires_grad(self.encoder, True)
        set_requires_grad(self.decoder, False)
        optimizer['encoder'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            restore_imgs = self.forward(real_imgs)

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size,
            loss_scaler=loss_scaler)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        loss_disc.backward()
        optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
            outputs = dict(
                log_vars=log_vars_disc,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        log_vars = {}
        log_vars.update(log_vars_encoder)

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    
    
# @MODELS.register_module()
# class AgileTransfer

# @MODELS.register_module()
# class AgileTranslation