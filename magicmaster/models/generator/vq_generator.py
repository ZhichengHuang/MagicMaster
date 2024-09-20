import torch
import torch.nn as nn
from mmengine.model import (BaseModule, normal_init, update_init_info,
                            xavier_init)
from magicmaster.registry import MODELS
from ..utils import get_module_device


@MODELS.register_module()
class VQGenerator(BaseModule):
    def __init__(self,
                 encoder,
                 post_encoder,
                 quantizer,
                 pre_decoder,
                 decoder):
        super().__init__()


        self.encoder = MODELS.build(encoder)

        self.post_encoder = MODELS.build(post_encoder)

        self.quantizer = MODELS.build(quantizer)

        self.pre_decoder = MODELS.build(pre_decoder)

        self.decoder = MODELS.build(decoder)



    def quantizer_image(self,image):
        z = self.post_encoder(self.encoder(image))

        codes,indice = self.quantizer(z)
        return codes,indice


    def decoder_image(self,codes):
        pred_image = self.decoder(self.pre_decoder(codes))
        return pred_image


    def forward(self,image):
        codes,indice = self.quantizer_image(image)
        pred_image = self.decoder_image(codes)
        return pred_image,codes,indice




