from .vqgan import *
from .magvit_v2 import (MagvitV2encoder,MagvitV2Adadecoder)
from .magvit_v2_2d import MagvitV2encoder2D, MagvitV2Adadecoder2D

__all__=[
    'VQGANEncoder', 'VQGANDecoder', 'VQGANAdaDecoder','MagvitV2encoder',
    'MagvitV2Adadecoder', 'MagvitV2encoder2D', 'MagvitV2Adadecoder2D'
]