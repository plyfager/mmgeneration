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
from collections import OrderedDict
import torch.distributed as dist
import torch.nn.functional as F

@MODELS.register_module()
class AgileEncoder(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 id_loss=None, 
                 perceptual_loss=None,
                 kl_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
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
            
        ## loss settings
        self.rec_loss = nn.MSELoss()
        if id_loss is not None:
            self.id_loss = build_module(id_loss)
        if perceptual_loss is not None:
            self.perceptual_loss = build_module(perceptual_loss)
        if kl_loss is not None:
            self.kl_loss = build_module(kl_loss)
        
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
    
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # Allow setting None for some loss item.
            # This is to support dynamic loss module, where the loss is
            # calculated with a fixed frequency.
            elif loss_value is None:
                continue
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # Note that you have to add 'loss' in name of the items that will be
        # included in back propagation.
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    def _get_encoder_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        
        img_gen = outputs_dict["restore_imgs"]
        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                      width // factor, factor)
            img_gen = img_gen.mean([3, 5])
        # inversion loss
        losses_dict['rec_loss'] = self.rec_loss(outputs_dict["real_imgs"], img_gen)
        losses_dict['id_loss'] = self.id_loss(outputs_dict["real_imgs"], img_gen)
        # losses_dict['perceptual_loss'] = self.perceptual_loss(outputs_dict["real_imgs"],outputs_dict["restore_imgs"])
        # losses_dict['kl_loss'] = self.kl_loss(outputs_dict["logvar"],outputs_dict["mu"])

        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

    def forward(self, x):
        code, logvar, mu = self.encoder(x)
        code = code.unbind(dim=1)
        rec_x = self.decoder(code, input_is_latent=True)
        return rec_x, logvar, mu
    
    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
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
        restore_imgs, logvar, mu = self.forward(real_imgs)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            encoder=self.encoder,
            decoder=self.decoder,
            logvar=logvar,
            mu=mu,
            restore_imgs=restore_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

        loss_encoder, log_vars_encoder = self._get_encoder_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_encoder))

        loss_encoder.backward()
        optimizer['encoder'].step()

        # Add downsampled images
        downsample_imgs = F.interpolate(restore_imgs, (256,256))
        
        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                downsample_imgs=downsample_imgs.cpu(),
                restore_imgs=restore_imgs.cpu(), real_imgs=real_imgs.cpu())
            outputs = dict(
                log_vars=log_vars_encoder,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        log_vars = {}
        log_vars.update(log_vars_encoder)

        results = dict(downsample_imgs=downsample_imgs.cpu(),restore_imgs=restore_imgs.cpu(), real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    
    
# @MODELS.register_module()
# class AgileTransfer

# @MODELS.register_module()
# class AgileTranslation