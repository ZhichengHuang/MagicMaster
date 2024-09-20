import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import (BaseModule, normal_init, update_init_info,
                            xavier_init)
from magicmaster.registry import MODELS
from typing import Optional


@MODELS.register_module()
class MagVitGenerator(BaseModule):
    def __init__(self,
                 encoder,
                 quantizer,
                 decoder):
        super().__init__()


        self.encoder = MODELS.build(encoder)


        self.quantizer = MODELS.build(quantizer)

        self.decoder = MODELS.build(decoder)



    def quantizer_image(self,video_or_images: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        x, cond,video_contains_first_frame = self.encoder(video_or_images, cond, video_contains_first_frame)

        codes,indice = self.quantizer(x)
        return codes,indice,cond,video_contains_first_frame


    def decoder_image(self,codes,cond, video_contains_first_frame):
        pred_image = self.decoder(codes,cond, video_contains_first_frame)
        return pred_image
    def forward(self,video_or_images: Tensor, cond: Optional[Tensor] = None, video_contains_first_frame=True):
        codes,indice,cond,video_contains_first_frame = self.quantizer_image(video_or_images, cond, video_contains_first_frame)
        recon_video = self.decoder_image(codes,cond, video_contains_first_frame)
        return recon_video,codes,indice




