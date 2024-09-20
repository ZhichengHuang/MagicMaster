import torch
import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn import Module, ModuleList

from typing import  Union, Tuple, Optional


from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from magicmaster.registry import MODELS



def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d




def divisible_by(num, den):
    return (num % den) == 0




def is_odd(n):
    return not divisible_by(n, 2)



def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)


def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)







class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        self.act = None

        # if act_fn is None:
        #     self.act = None
        # else:
        #     self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = cond
        if self.act:
            emb = self.act(emb)
        emb = torch.mean(emb, dim=(2, 3, 4), keepdim=False)
        emb = self.linear(emb)
        emb = emb[:, :, None, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x





class SpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = CausalConv3d(dim, dim_out * 4, [3,3,3],stride=[1,1,1])

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b c t h w -> b t c h w'),
            Rearrange('b t (c p1 p2) h w -> b t c (h p1) (w p2)', p1 = 2, p2 = 2),
            Rearrange('b t c h w -> b c t h w')
        )

        self.init_conv_(conv.conv)

    def init_conv_(self, conv):
        o, i,t, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i,t, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):

        out = self.net(x)

        return out

class TimeSpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = CausalConv3d(dim, dim_out * 8, [3,3,3],stride=[1,1,1])

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p1 p2 p3 ) t h w -> b c (t p1) (h p2) (w p3)', p1 = 2, p2 = 2, p3=2),
        )

        self.init_conv_(conv.conv)

    def init_conv_(self, conv):
        o, i,t, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 8, i, t, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 8) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):

        out = self.net(x)

        return out

def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    padding = [k // 2 for k in kernel_size]
    return nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding)




class CausalConv3d(Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride=[1,1,1],
        pad_mode = 'constant',
        **kwargs
    ):
        super().__init__()
        # kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride_time = stride[0]

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride_time)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        # stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        x = F.pad(x, self.time_causal_padding, mode = pad_mode)
        return self.conv(x)


class Residual(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        kernel_size=[3,3,3],
        pad_mode: str = 'constant',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._residual = nn.Sequential(
            nn.GroupNorm(32,in_channels,1e-6),
            nn.SiLU(),
            CausalConv3d(in_channels, out_channels, kernel_size, pad_mode = pad_mode),
            nn.GroupNorm(32,out_channels,
                1e-6,
            ),
            nn.SiLU(),
            CausalConv3d(out_channels, out_channels, kernel_size, pad_mode = pad_mode),
        )
        self._shortcut = (
            nn.Identity() if in_channels == out_channels else
            CausalConv3d(in_channels, out_channels, [1,1,1], pad_mode = pad_mode)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shortcut(x) + self._residual(x)


class Downsample(nn.Module):

    def __init__(self, num_channels: int,with_time=False) -> None:
        super().__init__()
        self._num_channels = num_channels
        stride=[1,1,1]
        if with_time:
            stride = [2, 2, 2]
        else:
            stride = [1, 2, 2]

        self._conv = CausalConv3d(
            num_channels,
            num_channels,
            kernel_size=[3,3,3],
            stride = stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        return x


@MODELS.register_module()
class MagvitV2encoder(Module):
    def __init__(self,
                 image_size,
                 channels=3,
                 init_dim=128,
                 layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                        ('consecutive_residual', 4),
                        ('spatial_down', 1),
                        ('channel_residual', 1),
                        ('consecutive_residual', 3),
                        ('time_spatial_down', 1),
                        ('consecutive_residual', 4),
                        ('time_spatial_down', 1),
                        ('channel_residual', 1),
                        ('consecutive_residual', 3),
                        ('consecutive_residual', 4),
                    ),
                 input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),
                 output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 pad_mode: str = 'constant',
                 separate_first_frame_encoding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # initial encoder

        self.conv_in = CausalConv3d(channels, init_dim, input_conv_kernel_size, pad_mode=pad_mode)

        # whether to encode the first frame separately or not

        self.conv_in_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(channels, init_dim, input_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        # encoder and decoder layers

        self.encoder_layers = ModuleList([])

        # self.conv_out = CausalConv3d(init_dim, channels, output_conv_kernel_size, pad_mode=pad_mode)

        dim = init_dim

        time_downsample_factor = 1
        self.has_cond_across_layers=[]

        for layer_def in layers:
            has_cond=False
            layer_type, *layer_params = cast_tuple(layer_def)


            if layer_type == 'consecutive_residual':
                num_consecutive, = layer_params
                encoder_layer = Sequential(
                    *[Residual(in_channels=dim, out_channels=dim) for _ in range(num_consecutive)])

            elif layer_type == 'spatial_down':
                encoder_layer = Downsample(dim,with_time=False)

            elif layer_type == 'channel_residual':
                num_consecutive, = layer_params
                encoder_layer = Residual(in_channels=dim, out_channels=dim*2)
                dim = dim*2

            elif layer_type == 'time_spatial_down':
                encoder_layer = Downsample(dim, with_time=True)

                time_downsample_factor *= 2

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            self.encoder_layers.append(encoder_layer)
            self.has_cond_across_layers.append(has_cond)


        layer_fmap_size = image_size


        # add a final norm just before quantization layer

        self.encoder_layers.append(Sequential(
            nn.GroupNorm(32, dim,1e-6),
                     nn.SiLU(),
            nn.Conv3d(dim,dim,[1,1,1],stride=[1,1,1])
        ))

        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size


    def encode(self, video: Tensor, cond: Optional[Tensor]=None,video_contains_first_frame=True):
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # whether to pad video or not

        if video_contains_first_frame:
            video_len = video.shape[2]

            video = pad_at_dim(video, (self.time_padding, 0), value=0., dim=2)

            video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, 'b c * h w')
            first_frame = self.conv_in_first_frame(first_frame)

        video = self.conv_in(video)

        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], 'b c * h w')
            video = pad_at_dim(video, (self.time_padding, 0), dim=2)

        # encoder layers

        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):
            layer_kwargs = dict()

            video = fn(video, **layer_kwargs)

        return video

    def forward(self,video_or_images: Tensor,
                cond: Optional[Tensor] = None,
                video_contains_first_frame = True,):
        assert video_or_images.ndim in {4, 5}

        assert video_or_images.shape[-2:] == (self.image_size, self.image_size)

        is_image = video_or_images.ndim == 4

        if is_image:
            video = rearrange(video_or_images, 'b c ... -> b c 1 ...')
            video_contains_first_frame = True

        else:
            video = video_or_images

        batch, channels, frames = video.shape[:3]

        assert divisible_by(frames - int(video_contains_first_frame),
                            self.time_downsample_factor), f'number of frames {frames} minus the first frame ({frames - int(video_contains_first_frame)}) must be divisible by the total downsample factor across time {self.time_downsample_factor}'

        # encoder
        x = self.encode(video, cond=cond, video_contains_first_frame=video_contains_first_frame)

        return x, cond,video_contains_first_frame



@MODELS.register_module()
class MagvitV2Adadecoder(Module):
    def __init__(self,
                 image_size,
                 channels=3,
                 init_dim=128,
                 layers: Tuple[Union[str, Tuple[str, int]], ...] = (
                        'residual',
                        'residual',
                        'residual'
                    ),
                 output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 separate_first_frame_encoding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        # initial encoder

        # whether to encode the first frame separately or not
        self.conv_out_first_frame = nn.Identity()

        if separate_first_frame_encoding:
            self.conv_out_first_frame = SameConv2d(init_dim, channels, output_conv_kernel_size[-2:])

        self.separate_first_frame_encoding = separate_first_frame_encoding

        self.decoder_layers = ModuleList([])


        dim = init_dim
        dim_out = dim

        layer_fmap_size = image_size
        time_downsample_factor = 1
        has_cond_across_layers = []

        for layer_def in layers:
            layer_type, *layer_params = cast_tuple(layer_def)

            if layer_type == 'consecutive_residual':
                has_cond = False
                num_consecutive, = layer_params
                decoder_layer = Sequential(
                    *[Residual(in_channels=dim, out_channels=dim) for _ in range(num_consecutive)])

            elif layer_type == 'spatial_up':
                has_cond = False
                decoder_layer = SpatialUpsample2x(dim)

            elif layer_type == 'channel_residual':
                has_cond = False
                num_consecutive, = layer_params
                decoder_layer = Residual(in_channels=dim* 2, out_channels=dim )
                dim = dim*2


            elif layer_type == 'time_spatial_up':
                has_cond = False
                decoder_layer = TimeSpatialUpsample2x(dim)

                time_downsample_factor *= 2

            elif layer_type =='condation':
                has_cond = True
                decoder_layer = AdaGroupNorm(embedding_dim=init_dim*4, out_dim=dim, num_groups=32)

            else:
                raise ValueError(f'unknown layer type {layer_type}')

            # self.decoder_layers.append(encoder_layer)

            self.decoder_layers.insert(0, decoder_layer)
            has_cond_across_layers.append(has_cond)
        self.decoder_layers.append(nn.GroupNorm(32, init_dim,1e-6),)
        self.decoder_layers.append(nn.SiLU(), )


        # self.conv_out = Sequential(
        #     nn.GroupNorm(32, init_dim,1e-6),
        #              nn.SiLU(),
        #     nn.Conv3d(init_dim,channels,[3,3,3],stride=[1,1,1]))
        self.conv_out = CausalConv3d(init_dim,channels,[1,1,1],stride=[1,1,1])

        # self.conv_in = CausalConv3d(512, 512, [3, 3, 3], stride=[1, 1, 1])


        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1

        self.fmap_size = layer_fmap_size

        # use a MLP stem for conditioning, if needed

        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)




    def decode(self,quantized: Tensor,cond: Optional[Tensor] = None,video_contains_first_frame = True):
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        batch = quantized.shape[0]

        #conditioning if needed



        x = quantized
        # x = self.conv_in(x)

        for fn, has_cond, in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):
            layer_kwargs = dict()

            if has_cond:
                layer_kwargs['cond']=quantized

            x = fn(x, **layer_kwargs)

        # to pixels
        if decode_first_frame_separately:
            left_pad, xff, x = x[:, :, :self.time_padding], x[:, :, self.time_padding], x[:, :,
                                                                                        (self.time_padding + 1):]
            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)

            video, _ = pack([outff, out], 'b c * h w')
        else:
            video = self.conv_out(x)

            # if video were padded, remove padding
            if video_contains_first_frame:
                video = video[:, :, self.time_padding:]

        return video

    def forward(self,quantized: Tensor,
                cond: Optional[Tensor] = None,
                video_contains_first_frame = True,):

        # decode
        recon_video = self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)

        return recon_video