# Copyright (c) OpenMMLab. All rights reserved.
from .clip_loss_comps import CLIPLossComps
from .disc_auxiliary_loss_comps import (DiscShiftLossComps,
                                        GradientPenaltyLossComps,
                                        R1GradientPenaltyComps)

from .gan_loss_comps import GANLossComps
from .gen_auxiliary_loss_comps import GeneratorPathRegularizerComps

__all__ = [
    'CLIPLossComps', 'DiscShiftLossComps', 'GradientPenaltyLossComps',
    'R1GradientPenaltyComps', 'GANLossComps',
    'GeneratorPathRegularizerComps'
]
