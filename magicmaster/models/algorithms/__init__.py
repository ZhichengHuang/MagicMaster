from .base_gan import *
from .base_vq_gan import *
from .average_model import ExponentialMovingAverage, RampUpEMA
from .magvit_vq_gan import MagVitVQGAN,MagVitAFVQGAN

__all__=[
    'VQGAN', 'ExponentialMovingAverage','RampUpEMA', 'MagVitVQGAN','MagVitAFVQGAN'
]