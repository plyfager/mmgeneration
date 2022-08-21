# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmgen.models.architectures.stylegan.modules import (MappingNetwork,
                                                         SynthesisInput,
                                                         SynthesisLayer,
                                                         SynthesisNetwork)


class TestMappingNetwork:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            noise_size=4,
            c_dim=0,
            style_channels=4,
            num_ws=2,
            num_layers=2,
            lr_multiplier=0.01,
            w_avg_beta=0.998)

    def test_cpu(self):
        module = MappingNetwork(**self.default_cfg)
        z = torch.randn([1, 4])
        c = None
        y = module(z, c)
        assert y.shape == (1, 2, 4)

        # test update_emas
        y = module(z, c, update_emas=True)
        assert y.shape == (1, 2, 4)

        # test truncation
        y = module(z, c, truncation=2)
        assert y.shape == (1, 2, 4)

        # test with c_dim>0
        cfg = deepcopy(self.default_cfg)
        cfg.update(c_dim=2)
        module = MappingNetwork(**cfg)
        z = torch.randn([2, 4])
        c = torch.eye(2)
        y = module(z, c)
        assert y.shape == (2, 2, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = MappingNetwork(**self.default_cfg).cuda()
        z = torch.randn([1, 4]).cuda()
        c = None
        y = module(z, c)
        assert y.shape == (1, 2, 4)

        # test update_emas
        y = module(z, c, update_emas=True).cuda()
        assert y.shape == (1, 2, 4)

        # test truncation
        y = module(z, c, truncation=2).cuda()
        assert y.shape == (1, 2, 4)

        # test with c_dim>0
        cfg = deepcopy(self.default_cfg)
        cfg.update(c_dim=2)
        module = MappingNetwork(**cfg).cuda()
        z = torch.randn([2, 4]).cuda()
        c = torch.eye(2).cuda()
        y = module(z, c)
        assert y.shape == (2, 2, 4)


class TestSynthesisInput:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            style_channels=6,
            channels=4,
            size=8,
            sampling_rate=16,
            bandwidth=2)

    def test_cpu(self):
        module = SynthesisInput(**self.default_cfg)
        x = torch.randn((2, 6))
        y = module(x)
        assert y.shape == (2, 4, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = SynthesisInput(**self.default_cfg).cuda()
        x = torch.randn((2, 6)).cuda()
        y = module(x)
        assert y.shape == (2, 4, 8, 8)


class TestSynthesisLayer:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            style_channels=6,
            is_torgb=False,
            is_critically_sampled=False,
            use_fp16=False,
            conv_kernel=3,
            in_channels=3,
            out_channels=3,
            in_size=16,
            out_size=16,
            in_sampling_rate=16,
            out_sampling_rate=16,
            in_cutoff=2,
            out_cutoff=2,
            in_half_width=6,
            out_half_width=6)

    def test_cpu(self):
        module = SynthesisLayer(**self.default_cfg)
        x = torch.randn((2, 3, 16, 16))
        w = torch.randn((2, 6))
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)

        # test update_emas
        y = module(x, w, update_emas=True)
        assert y.shape == (2, 3, 16, 16)

        # test force_fp32
        cfg = deepcopy(self.default_cfg)
        cfg.update(use_fp16=True)
        module = SynthesisLayer(**cfg)
        x = torch.randn((2, 3, 16, 16))
        w = torch.randn((2, 6))
        y = module(x, w, force_fp32=False)
        assert y.shape == (2, 3, 16, 16)
        assert y.dtype == torch.float32

        # test critically_sampled
        cfg = deepcopy(self.default_cfg)
        cfg.update(is_critically_sampled=True)
        module = SynthesisLayer(**cfg)
        x = torch.randn((2, 3, 16, 16))
        w = torch.randn((2, 6))
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = SynthesisLayer(**self.default_cfg).cuda()
        x = torch.randn((2, 3, 16, 16)).cuda()
        w = torch.randn((2, 6)).cuda()
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)

        # test update_emas
        y = module(x, w, update_emas=True).cuda()
        assert y.shape == (2, 3, 16, 16)

        # test critically_sampled
        cfg = deepcopy(self.default_cfg)
        cfg.update(is_critically_sampled=True)
        module = SynthesisLayer(**cfg).cuda()
        x = torch.randn((2, 3, 16, 16)).cuda()
        w = torch.randn((2, 6)).cuda()
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)


class TestSynthesisNetwork:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            style_channels=8, out_size=16, img_channels=3, num_layers=4)

    def test_cpu(self):
        module = SynthesisNetwork(**self.default_cfg)
        ws = torch.randn((2, 6, 8))
        y = module(ws)
        assert y.shape == (2, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = SynthesisNetwork(**self.default_cfg).cuda()
        ws = torch.randn((2, 6, 8)).cuda()
        y = module(ws)
        assert y.shape == (2, 3, 16, 16)
