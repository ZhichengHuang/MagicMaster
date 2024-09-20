import torch
import torch.nn as nn
import torch.nn.functional as F
from magicmaster.registry import MODELS

@MODELS.register_module()
class ConvConnector(nn.Module):
    def __init__(self,in_channels=256, out_channels=256):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )
    def forward(self,x):
        return self._conv(x)

