# Copyright (c) OpenMMLab. All rights reserved.
from .disc_auxiliary_loss import (DiscShiftLoss, GradientPenaltyLoss,
                                  R1GradientPenalty, disc_shift_loss,
                                  gradient_penalty_loss,
                                  r1_gradient_penalty_loss)
from .gan_loss import GANLoss
from .gen_auxiliary_loss import GeneratorPathRegularizer, gen_path_regularizer
from .id_loss import IDLoss
from .kl_loss import KLloss
from .pixelwise_loss import L1Loss, MSELoss, gaussian_kld
from .criteria.lpips.lpips import LPIPS

__all__ = [
    'GANLoss', 'DiscShiftLoss', 'disc_shift_loss', 'gradient_penalty_loss',
    'GradientPenaltyLoss', 'R1GradientPenalty', 'r1_gradient_penalty_loss',
    'GeneratorPathRegularizer', 'gen_path_regularizer', 'MSELoss', 'L1Loss',
    'IDLoss', 'KLloss', 'gaussian_kld', 'LPIPS'
]
