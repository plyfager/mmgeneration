import os.path as osp
from collections import OrderedDict
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.builder import MODELS, build_module
from mmgen.models.misc import tensor2img
from ..common import set_requires_grad
from .static_unconditional_gan import StaticUnconditionalGAN


def downsample_256(img):
    batch, channel, height, width = img.shape
    if height > 256:
        factor = height // 256
        img = img.reshape(batch, channel, height // factor, factor,
                          width // factor, factor)
        img = img.mean([3, 5])
    return img


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
        self.fixed_mlp = self.decoder.style_mapping

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

        img_gen = outputs_dict['restore_imgs']
        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                      width // factor, factor)
            img_gen = img_gen.mean([3, 5])
        # inversion loss
        losses_dict['rec_loss'] = self.rec_loss(outputs_dict['real_imgs'],
                                                img_gen)
        losses_dict['id_loss'] = self.id_loss(outputs_dict['real_imgs'],
                                              img_gen)
        losses_dict['perceptual_loss'] = self.perceptual_loss(
            outputs_dict['real_imgs'], img_gen)
        losses_dict['kl_loss'] = self.kl_loss(outputs_dict['logvar'],
                                              outputs_dict['mu'])

        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

    def forward(self, x, test_mode=False):
        code, logvar, mu = self.encoder(x)
        if test_mode:
            code = mu
        w_plus_code = [self.fixed_mlp(s) for s in code]
        w_plus_code = [torch.stack(w_plus_code, dim=0)]
        rec_x = self.decoder(w_plus_code, input_is_latent=True)
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
        downsample_imgs = F.interpolate(restore_imgs, (256, 256))

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                downsample_imgs=downsample_imgs.cpu(),
                restore_imgs=restore_imgs.cpu(),
                real_imgs=real_imgs.cpu())
            outputs = dict(
                log_vars=log_vars_encoder,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        log_vars = {}
        log_vars.update(log_vars_encoder)

        results = dict(
            downsample_imgs=downsample_imgs.cpu(),
            restore_imgs=restore_imgs.cpu(),
            real_imgs=real_imgs.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs


@MODELS.register_module()
class AgileTransfer(StaticUnconditionalGAN):

    def __init__(self,
                 src_generator,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(
            generator,
            discriminator,
            gan_loss,
            disc_auxiliary_loss=disc_auxiliary_loss,
            gen_auxiliary_loss=gen_auxiliary_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        self.src_g_cfg = deepcopy(src_generator)
        self.source_generator = build_module(src_generator)
        self.perceptual_loss = build_module(perceptual_loss)
        set_requires_grad(self.source_generator, False)
        self.fixed_mlp = deepcopy(self.generator.style_mapping)

    def forward(self, x):
        return dict(
            source_result=self.source_generator(x, input_is_latent=True),
            target_result=self.generator(x, input_is_latent=True))

    def latent_generator(self, batch_size):
        z_plus_code = torch.randn(batch_size * 18, 512)
        w_plus_code = self.fixed_mlp(z_plus_code)
        w_plus_code = w_plus_code.view(-1, 18, 512)
        w_plus_code = w_plus_code.unbind(dim=1)
        return w_plus_code

    def _get_gen_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        # import ipdb
        # ipdb.set_trace()
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake_g'] = self.gan_loss(
            outputs_dict['disc_pred_fake_g'],
            target_is_real=True,
            is_disc=False)
        # TODO: add modified LPIPS
        source_results = self.source_generator(
            outputs_dict['latents'], input_is_latent=True)
        resized_source_results = downsample_256(source_results)
        resized_target_results = downsample_256(outputs_dict['fake_imgs'])
        losses_dict['loss_sim'] = self.perceptual_loss(
            gt=resized_source_results, x=resized_target_results)
        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue

                # mmcv.print_log(f'get loss for {loss_module.name()}')
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var, source_results

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data_batch (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            loss_scaler (:obj:`torch.cuda.amp.GradScaler` | None, optional):
                The loss/gradient scaler used for auto mixed-precision
                training. Defaults to ``None``.
            use_apex_amp (bool, optional). Whether to use apex.amp. Defaults to
                ``False``.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
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

        # disc training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling
        with torch.no_grad():
            fake_imgs = self.generator(None, num_batches=batch_size)

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

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_disc, optimizer['discriminator'],
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['discriminator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['discriminator'])
            # loss_scaler.update will be called in runner.train()
        else:
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

        # generator training
        set_requires_grad(self.discriminator, False)
        optimizer['generator'].zero_grad()

        # TODO: add noise sampler to customize noise sampling
        latents = self.latent_generator(batch_size)
        fake_imgs = self.generator(latents, input_is_latent=True)
        disc_pred_fake_g = self.discriminator(fake_imgs)

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_imgs,
            disc_pred_fake_g=disc_pred_fake_g,
            iteration=curr_iter,
            batch_size=batch_size,
            loss_scaler=loss_scaler,
            latents=latents)

        loss_gen, log_vars_g, source_results = self._get_gen_loss(
            data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        if loss_scaler:
            loss_scaler.scale(loss_gen).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_gen, optimizer['generator'],
                    loss_id=1) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_gen.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['generator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['generator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['generator'].step()
        # self.generator.module.style_mapping.load_state_dict(self.fixed_mlp.state_dict())

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)

        results = dict(
            fake_imgs=fake_imgs.cpu(),
            real_imgs=real_imgs.cpu(),
            src_g_imgs=source_results.cpu())
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs


# @MODELS.register_module()
# class AgileTranslation
