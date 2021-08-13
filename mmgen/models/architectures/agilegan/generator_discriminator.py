import torch.nn as nn
from .modules import get_blocks
from .modules import bottleneck_IR_SE, SubBlock
import torch.nn.functional as F
import torch
from mmgen.models.builder import MODULES

@MODULES.register_module()
class VAEStyleEncoder(nn.Module):
    def __init__(self, num_layers, input_nc=3,pretrained=None):
        super(VAEStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152]
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
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
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)


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
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

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
