from .agilegan import AgileEncoder, AgileTransfer
from .base_gan import BaseGAN
from .basic_conditional_gan import BasicConditionalGAN
from .mspie_stylegan2 import MSPIEStyleGAN2
from .progressive_growing_unconditional_gan import ProgressiveGrowingGAN
from .singan import PESinGAN, SinGAN
from .static_unconditional_gan import StaticUnconditionalGAN

__all__ = [
    'BaseGAN', 'StaticUnconditionalGAN', 'ProgressiveGrowingGAN', 'SinGAN',
<<<<<<< HEAD
    'Pix2Pix', 'CycleGAN', 'MSPIEStyleGAN2', 'PESinGAN', 'BasicConditionalGAN',
    'AgileEncoder', 'AgileTransfer'
=======
    'MSPIEStyleGAN2', 'PESinGAN', 'BasicConditionalGAN'
>>>>>>> bfd00c3b3de936554dc685d7cdc3cb32dc9d43ad
]
