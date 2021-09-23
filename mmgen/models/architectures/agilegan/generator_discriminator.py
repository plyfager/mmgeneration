import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

from mmgen.models.builder import MODULES
from .modules import SubBlock, bottleneck_IR_SE, get_blocks
from ..pggan import PixelNorm
from .styleganv2_modules import ConstantInput, EqualLinearActModule, ModulatedStyleConv, ModulatedToRGB

import numpy as np

@MODULES.register_module()
class VAEStyleEncoder(nn.Module):

    def __init__(self, num_layers, input_nc=3, pretrained=None):
        super(VAEStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152]
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = SubBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = SubBlock(512, 512, 32)
            else:
                style = SubBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(
            256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(
            128, 512, kernel_size=1, stride=1, padding=0)

        self.fc_mu = nn.Linear(512, 512)
        self.fc_var = nn.Linear(512, 512)

        self.fc_mu.weight.data.fill_(0)
        self.fc_mu.bias.data.fill_(0)

        self.fc_var.weight.data.fill_(0)
        self.fc_var.bias.data.fill_(0)
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmgen')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
        out = torch.stack(latents, dim=1)

        mu = self.fc_mu(out)
        logvar = self.fc_var(out)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, logvar, mu


@MODULES.register_module()
class DualGenerator(nn.Module):
    
    def __init__(self,
                 out_size,
                 style_channels,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 num_fp16_scales=0,
                 fp16_enabled=False,
                 pretrained=None):
        super().__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.num_mlps = num_mlps
        self.channel_multiplier = channel_multiplier
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob
        self.num_fp16_scales = num_fp16_scales
        self.fp16_enabled = fp16_enabled

        # define style mapping layers
        mapping_layers = [PixelNorm()]

        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.),
                    act_cfg=dict(type='fused_bias')))

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # constant input layer
        self.constant_input = ConstantInput(self.channels[4])
        # 4x4 stage
        self.conv1 = ModulatedStyleConv(
            self.channels[4],
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel)
        self.to_rgb1 = ModulatedToRGB(
            self.channels[4],
            style_channels,
            upsample=False,
            fp16_enabled=fp16_enabled)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convsA = nn.ModuleList()
        self.to_rgbsA = nn.ModuleList()

        self.convsB = nn.ModuleList()
        self.to_rgbsB = nn.ModuleList()

        self.convsC = nn.ModuleList()
        self.to_rgbsC = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(3, self.log_size + 1):
            
            if i<7:
                #A
                out_channels_ = self.channels[2**i]

                # If `fp16_enabled` is True, all of layers will be run in auto
                # FP16. In the case of `num_fp16_sacles` > 0, only partial
                # layers will be run in fp16.
                _use_fp16 = (self.log_size - i) < num_fp16_scales or fp16_enabled

                self.convsA.append(
                    ModulatedStyleConv(
                        in_channels_,
                        out_channels_,
                        3,
                        style_channels,
                        upsample=True,
                        blur_kernel=blur_kernel,
                        fp16_enabled=_use_fp16))
                self.convsA.append(
                    ModulatedStyleConv(
                        out_channels_,
                        out_channels_,
                        3,
                        style_channels,
                        upsample=False,
                        blur_kernel=blur_kernel,
                        fp16_enabled=_use_fp16))
                self.to_rgbsA.append(
                    ModulatedToRGB(
                        out_channels_,
                        style_channels,
                        upsample=True,
                        fp16_enabled=_use_fp16))  # set to global fp16
                # B
                self.convsB.append(
                    ModulatedStyleConv(
                        in_channels_,
                        out_channels_,
                        3,
                        style_channels,
                        upsample=True,
                        blur_kernel=blur_kernel,
                        fp16_enabled=_use_fp16))
                self.convsB.append(
                    ModulatedStyleConv(
                        out_channels_,
                        out_channels_,
                        3,
                        style_channels,
                        upsample=False,
                        blur_kernel=blur_kernel,
                        fp16_enabled=_use_fp16))
                self.to_rgbsB.append(
                    ModulatedToRGB(
                        out_channels_,
                        style_channels,
                        upsample=True,
                        fp16_enabled=_use_fp16))  # set to global fp16 
            else:
                out_channels_ = self.channels[2**i]
                self.convsC.append(
                    ModulatedStyleConv(
                        in_channels_,
                        out_channels_,
                        3,
                        style_channels,
                        upsample=True,
                        blur_kernel=blur_kernel,
                        fp16_enabled=_use_fp16))
                self.convsC.append(
                    ModulatedStyleConv(
                        out_channels_,
                        out_channels_,
                        3,
                        style_channels,
                        upsample=False,
                        blur_kernel=blur_kernel,
                        fp16_enabled=_use_fp16))
                self.to_rgbsC.append(
                    ModulatedToRGB(
                        out_channels_,
                        style_channels,
                        upsample=True,
                        fp16_enabled=_use_fp16))  # set to global fp16
            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        for layer_idx in range(self.num_injected_noises):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 torch.randn(*shape))

        if pretrained is not None:
            print(pretrained)
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmgen')

    def train(self, mode=True):
        if mode:
            if self.default_style_mode != self._default_style_mode:
                mmcv.print_log(
                    f'Switch to train style mode: {self._default_style_mode}',
                    'mmgen')
            self.default_style_mode = self._default_style_mode

        else:
            if self.default_style_mode != self.eval_style_mode:
                mmcv.print_log(
                    f'Switch to evaluation style mode: {self.eval_style_mode}',
                    'mmgen')
            self.default_style_mode = self.eval_style_mode

        return super(DualGenerator, self).train(mode)

    def make_injected_noise(self):
        """make noises that will be injected into feature maps.

        Returns:
            list[Tensor]: List of layer-wise noise tensor.
        """
        device = get_module_device(self)

        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        return get_mean_latent(self, num_samples, **kwargs)

    def style_mixing(self,
                     n_source,
                     n_target,
                     inject_index=1,
                     truncation_latent=None,
                     truncation=0.7):
        return style_mixing(
            self,
            n_source=n_source,
            n_target=n_target,
            inject_index=inject_index,
            truncation=truncation,
            truncation_latent=truncation_latent,
            style_channels=self.style_channels)

    def forward(self,
                styles,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                injected_noise=None,
                randomize_noise=True):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """
        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmcv.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    noise_generator((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    torch.randn((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]

        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []
            # calculate truncation latent on the fly
            if truncation_latent is None and not hasattr(
                    self, 'truncation_latent'):
                self.truncation_latent = self.get_mean_latent()
                truncation_latent = self.truncation_latent
            elif truncation_latent is None and hasattr(self,
                                                       'truncation_latent'):
                truncation_latent = self.truncation_latent

            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # 4x4 stage
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        #
        outA, outB = out.clone(),out.clone()
        skipA, skipB = skip.clone(),skip.clone()        
        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convsA[::2], self.convsA[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbsA):
            outA = up_conv(outA, latent[:, _index], noise=noise1)
            outA = conv(outA, latent[:, _index + 1], noise=noise2)
            skipA = to_rgb(outA, latent[:, _index + 2], skipA)
            _index += 2
        
        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convsB[::2], self.convsB[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbsB):
            outB = up_conv(outB, latent[:, _index], noise=noise1)
            outB = conv(outB, latent[:, _index + 1], noise=noise2)
            skipB = to_rgb(outB, latent[:, _index + 2], skipB)
            _index += 2

        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convsC[::2], self.convsC[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbsC):
            outA = up_conv(outA, latent[:, _index], noise=noise1)
            outA = conv(outA, latent[:, _index + 1], noise=noise2)
            skipA = to_rgb(outA, latent[:, _index + 2], skipA)
            
            outB = up_conv(outB, latent[:, _index], noise=noise1)
            outB = conv(outB, latent[:, _index + 1], noise=noise2)
            skipB = to_rgb(outB, latent[:, _index + 2], skipB)
            _index += 2
        # make sure the output image is torch.float32 to avoid RunTime Error
        # in other modules
        imgA = skipA.to(torch.float32)
        imgB = skipB.to(torch.float32)

        if return_latents or return_noise:
            output_dict = dict(
                fake_imgA=imgA,
                fake_imgB=imgB,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch)
            return output_dict

        return imgA, imgB

