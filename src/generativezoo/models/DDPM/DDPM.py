###############################################################################
######### Code based on: https://github.com/cloneofsimo/minDiffusion ##########
### https://github.com/DhruvSrikanth/DenoisingDiffusionProbabilisticModels  ###
#################  https://github.com/ermongroup/ddim ######################### 
###############################################################################

import torch.nn as nn
from einops import rearrange
from torch import einsum
import torch
import math
from functools import partial
import torch.nn.functional as F
import os
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
from config import models_dir
from torchvision.transforms import Compose, Lambda, ToPILImage
from torchvision.utils import make_grid
from lpips import LPIPS
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
from collections import OrderedDict
import copy
from abc import abstractmethod
import cv2

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'DDPM')):
    os.makedirs(os.path.join(models_dir, 'DDPM'))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.5):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # if name contains "module" then remove module
        if "module" in name:
            name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_head_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_head_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_head_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

def plot_samples(samples):
    '''
    Plot samples
    :param samples: samples to plot
    '''
    n_rows = int(np.sqrt(samples.shape[0]))
    n_cols = n_rows
    samples = samples * 0.5 + 0.5
    samples = np.clip(samples, 0, 1)
    grid = make_grid(torch.tensor(samples), nrow=n_rows, normalize=False, padding=0)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def plot_inpainting(original, masks, inpainted):
    '''
    Plot inpainting results
    :param original: original images
    :param masks: masks
    :param inpainted: inpainted images
    '''
    n_rows = int(np.sqrt(original.shape[0]))
    original = original * 0.5 + 0.5
    original = np.clip(original, 0, 1)
    inpainted = inpainted * 0.5 + 0.5
    inpainted = np.clip(inpainted, 0, 1)
    
    if masks.shape[1]!=original.shape[1]:
        masks = masks[:, :original.shape[1], :, :] # remove extra channel
        # upscale masks to shape[2] and shape[3]
        masks = F.interpolate(masks, size=(original.shape[2], original.shape[3]), mode='nearest')
    masks = masks.cpu().numpy()

    
    # multiply masks with original images
    masked = original * masks

    grid_og = make_grid(torch.tensor(masked), nrow=n_rows, normalize=False, padding=0)
    grid_inp = make_grid(torch.tensor(inpainted), nrow=n_rows, normalize=False, padding=0)
    # plot both side by side
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(grid_og.permute(1, 2, 0))
    ax[0].axis('off')
    ax[0].set_title('Original images with masks')
    ax[1].imshow(grid_inp.permute(1, 2, 0))
    ax[1].axis('off')
    ax[1].set_title('Inpainted images')
    plt.show()

class DDPM(nn.Module):
    def __init__(self, args, image_size, channels):
        '''
        DDPM module
        :param args: arguments
        :param image_size: size of the image
        :param in_channels: number of input channels
        '''
        super().__init__()
        self.reverse_transform = Compose([
                                        Lambda(lambda t: (t + 1) / 2),
                                        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
                                        Lambda(lambda t: t * 255.),
                                        Lambda(lambda t: t.numpy().astype(np.uint8)),
                                        ToPILImage(),
                                    ])
        
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(self.device) if args.latent else None
        self.channels = channels
        self.img_size = image_size

        # If using VAE, change the number of channels and image size accordingly
        if self.vae is not None:
            self.channels = 4
            self.img_size = self.img_size // 8

        self.model = UNetModel(
            image_size=self.img_size,
            in_channels=self.channels,
            model_channels=args.model_channels,
            out_channels=self.channels,
            num_res_blocks=args.num_res_blocks,
            attention_resolutions=args.attention_resolutions,
            dropout=args.dropout,
            channel_mult=args.channel_mult,
            conv_resample=args.conv_resample,
            dims=args.dims,
            num_classes=None,
            use_checkpoint=False,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=args.use_scale_shift_norm,
            resblock_updown=args.resblock_updown,
            use_new_attention_order=args.use_new_attention_order
        ).to(self.device)

        self.scheduler = LinearScheduler(args.beta_start, args.beta_end, args.timesteps)
        self.forward_diffusion_model = ForwardDiffusion(self.scheduler.sqrt_alphas_cumprod, self.scheduler.sqrt_one_minus_alphas_cumprod, self.reverse_transform)
        self.sampler = Sampler(self.scheduler.betas, args.timesteps, args.sample_timesteps, args.ddpm, args.recon_factor)
        self.criterion = get_loss
        self.n_epochs = args.n_epochs
        self.timesteps = args.timesteps
        self.sample_and_save_freq = args.sample_and_save_freq
        self.loss_type = args.loss_type
        self.dataset = args.dataset
        self.no_wandb = args.no_wandb
        self.lr = args.lr
        self.warmup = args.warmup
        self.decay = args.decay
        self.snapshot = args.n_epochs//args.snapshots # take the snapshot every x epochs

        if args.lpips:
            self.lpips_loss = LPIPS(net='alex').to(self.device)
        else:
            self.lpips_loss = None
        
        if args.train:
            self.ema = copy.deepcopy(self.model)
            self.ema_rate = args.ema_rate
            for param in self.ema.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        '''
        Encode the input image
        :param x: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.encode(x)
        else:
            return self.vae.encode(x)
        
    @torch.no_grad()    
    def decode(self, z):
        '''
        Decode the input image
        :param z: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.decode(z)
        else:
            return self.vae.decode(z)
    


    def train_model(self, dataloader, verbose=True):
        '''
        Train the model
        :param dataloader: dataloader
        '''

        accelerate = Accelerator(log_with="wandb")
        if not self.no_wandb:
            accelerate.init_trackers(project_name='DDPM',
            config = {
                        'dataset': self.args.dataset,
                        'batch_size': self.args.batch_size,
                        'n_epochs': self.args.n_epochs,
                        'lr': self.args.lr,
                        'timesteps': self.args.timesteps,
                        'beta_start': self.args.beta_start,
                        'beta_end': self.args.beta_end,
                        'ddpm': self.args.ddpm,
                        'input_size': self.img_size,
                        'channels': self.channels,
                        'loss_type': self.args.loss_type,
                        'model_channels': self.args.model_channels,
                        'num_res_blocks': self.args.num_res_blocks,
                        'attention_resolutions': self.args.attention_resolutions,
                        'dropout': self.args.dropout,
                        'channel_mult': self.args.channel_mult,
                        'conv_resample': self.args.conv_resample,
                        'dims': self.args.dims,
                        'num_heads': self.args.num_heads,
                        'num_head_channels': self.args.num_head_channels,
                        'use_scale_shift_norm': self.args.use_scale_shift_norm,
                        'resblock_updown': self.args.resblock_updown,
                        'use_new_attention_order': self.args.use_new_attention_order, 
                        "ema_rate": self.args.ema_rate,
                        "warmup": self.args.warmup,
                        "latent": self.args.latent,
                        "decay": self.args.decay,
                        "size": self.args.size,   
                },
                init_kwargs={"wandb":{"name": f"DDPM_{self.args.dataset}"}})

        best_loss = np.inf
        create_checkpoint_dir()
        epoch_bar = tqdm(range(self.n_epochs), desc='Epochs', leave=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.n_epochs*len(dataloader), pct_start=self.warmup/self.n_epochs, anneal_strategy='cos', cycle_momentum=False, div_factor=self.lr/1e-6, final_div_factor=1)

        if self.vae is None:
            dataloader, self.model, optimizer, scheduler, self.ema = accelerate.prepare(dataloader, self.model, optimizer, scheduler, self.ema)
        else:
            dataloader, self.model, optimizer, scheduler, self.ema, self.vae = accelerate.prepare(dataloader, self.model, optimizer, scheduler, self.ema, self.vae) 

        update_ema(self.ema, self.model, 0)


        for epoch in epoch_bar:
            self.model.train()
            acc_loss = 0.0
            for batch in tqdm(dataloader, desc=f'Batches', leave=False, disable=not verbose):
                optimizer.zero_grad()
                batch_size = batch[0].shape[0]
                batch = batch[0].to(self.device)

                with accelerate.autocast():
                    if self.vae is not None:
                        with torch.no_grad():
                            # if x has one channel, make it 3 channels
                            if batch.shape[1] == 1:
                                batch = torch.cat((batch, batch, batch), dim=1)
                            batch = self.encode(batch).latent_dist.sample().mul_(0.18215)

                    t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                    loss = self.criterion(forward_diffusion_model=self.forward_diffusion_model, denoising_model=self.model, x_start=batch, t=t, loss_type=self.loss_type)
                    accelerate.backward(loss)
                
                optimizer.step()
                scheduler.step()
                acc_loss += loss.item() * batch_size
                update_ema(self.ema, self.model, self.ema_rate)
            
            accelerate.wait_for_everyone()

            if not self.no_wandb:
                accelerate.log({"Train Loss": acc_loss / len(dataloader.dataset)})
                accelerate.log({"Learning Rate": scheduler.get_last_lr()[0]})
            
            epoch_bar.set_postfix({'Loss': acc_loss/len(dataloader.dataset)})   

            # save generated images
            if epoch == 0 or (epoch+1) % self.sample_and_save_freq == 0:
                samples = self.sampler.sample(model=self.ema, image_size=self.img_size, batch_size=16, channels=self.channels)
                all_images = torch.tensor(samples[-1])

                if self.vae is not None:
                    with torch.no_grad():
                        all_images = self.decode(all_images.to(self.device) / 0.18215).sample 
                        all_images = all_images.cpu().detach()

                all_images = all_images * 0.5 + 0.5
                all_images = all_images.clamp(0, 1)
                fig = plt.figure(figsize=(10, 10))
                grid = make_grid(all_images, nrow=int(np.sqrt(all_images.shape[0])), normalize=False, padding=0)
                plt.imshow(grid.permute(1, 2, 0))
                plt.xticks([])
                plt.yticks([])
                
                #save figure wandb
                if not self.no_wandb:
                    accelerate.log({"DDPM Samples": fig})
                plt.close(fig)

            if acc_loss/len(dataloader.dataset) < best_loss:
                best_loss = acc_loss/len(dataloader.dataset)

            if (epoch+1) % self.snapshot == 0:
                #torch.save(self.ema.state_dict(), os.path.join(models_dir,'DDPM',f"{'LatDDPM' if self.vae is not None else 'DDPM'}_{self.dataset}.pt"))
                ema_to_save = accelerate.unwrap_model(self.ema)
                accelerate.save(ema_to_save.state_dict(), os.path.join(models_dir,'DDPM',f"{'LatDDPM' if self.vae is not None else 'DDPM'}_{self.dataset}_epoch{epoch+1}.pt"))
    
    @torch.no_grad()
    def outlier_score(self, x_start):
        '''
        Compute the outlier score
        :param x_start: input image
        '''
        if self.vae is not None:
            with torch.no_grad():
                # if x has one channel, make it 3 channels
                if x_start.shape[1] == 1:
                    x_start = torch.cat((x_start, x_start, x_start), dim=1)
                x_original = x_start.clone()
                x_start = self.encode(x_start).latent_dist.sample().mul_(0.18215)

        noise = torch.randn_like(x_start)
        t = torch.ones((x_start.shape[0],), device=self.device).long() * (int(self.timesteps*self.sampler.recon_factor)-1)

        x_noisy = self.forward_diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_image = self.sampler.reconstruct_loop(self.model, self.img_size, x_noisy)[-1]
        predicted_image = torch.tensor(predicted_image, device=self.device)

        if self.vae is not None:
            with torch.no_grad():
                predicted_image = self.decode(predicted_image.to(self.device) / 0.18215).sample 
                predicted_image = predicted_image.cpu().detach()
                x_start = x_original

        if self.loss_type == 'l1':
            loss = nn.L1Loss(reduction = 'none')
            elementwise_loss = torch.mean(loss(x_start, predicted_image).reshape(x_start.shape), dim=(1,2,3))
        elif self.loss_type == 'l2':
            loss = nn.MSELoss(reduction = 'none')
            elementwise_loss = torch.mean(loss(x_start, predicted_image).reshape(x_start.shape), dim=(1,2,3))
        elif self.loss_type == "huber":
            loss = nn.HuberLoss(reduction = 'none')
            elementwise_loss = torch.mean(loss(x_start, predicted_image).reshape(x_start.shape), dim=(1,2,3))
        else:
            raise NotImplementedError()
        
        if self.lpips_loss is not None:
            # if image only has 1 channel, repeat it to 3 channels
            if predicted_image.shape[1] == 1:
                predicted_image = predicted_image.repeat(1,3,1,1)
                x_start = x_start.repeat(1,3,1,1)

            lpips = torch.mean(self.lpips_loss(predicted_image, x_start), dim=(1,2,3))
            elementwise_loss += lpips

        return elementwise_loss
    
    @torch.no_grad()
    def outlier_detection(self, val_loader, out_loader, in_name, out_name):
        '''
        Outlier detection
        :param val_loader: validation loader
        :param out_loader: outlier loader
        :param in_name: name of the in-distribution dataset
        :param out_name: name of the out-of-distribution dataset
        '''
        self.model.eval()
        val_loss = 0.0
        val_scores = []
        for batch in tqdm(val_loader, desc='In-distribution', leave=True):
            batch = batch[0].to(self.device)
            score = self.outlier_score(x_start=batch)
            val_scores.append(score.cpu().numpy())
        val_scores = np.concatenate(val_scores)

        out_scores = []

        for batch in tqdm(out_loader, desc='Out-of-distribution', leave=True):
            batch = batch[0].to(self.device)
            score = self.outlier_score(x_start=batch)
            out_scores.append(score.cpu().numpy())
        out_scores = np.concatenate(out_scores)
        
        y_true = np.concatenate([np.zeros_like(val_scores), np.ones_like(out_scores)], axis=0)
        y_score = np.concatenate([val_scores, out_scores], axis=0)
        auc_score = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve([0]*len(val_scores) + [1]*len(out_scores), np.concatenate([val_scores, out_scores]))
        fpr95 = fpr[np.argmax(tpr >= 0.95)]

        print('AUC score: {:.5f}'.format(auc_score), 'FPR95: {:.5f}'.format(fpr95))
        plt.figure(figsize=(10, 6))
        plt.hist(val_scores, bins=100, alpha=0.5, label='In-distribution')
        plt.hist(out_scores, bins=100, alpha=0.5, label='Out-of-distribution')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        # legend top right corner
        plt.legend(loc='upper right')
        plt.title('{} vs {} AUC: {:.2f} FPR95: {:.2f}'.format(in_name, out_name, 100*auc_score, 100*fpr95))
        plt.show()

    def create_masks(self, batch_size):
        '''
        Create masks for inpainting
        :param batch_size: batch size
        '''
        masks = torch.ones((batch_size, self.channels, self.img_size, self.img_size), device=self.device)
        startx = torch.randint(0, self.img_size//2, (batch_size,), device=self.device)
        starty = torch.randint(0, self.img_size//2, (batch_size,), device=self.device)
        endx = startx + torch.randint(self.img_size//4, self.img_size//2, (batch_size,), device=self.device)
        endy = starty + torch.randint(self.img_size//4, self.img_size//2, (batch_size,), device=self.device)
        for i in range(batch_size):
            masks[i, :, startx[i]:endx[i], starty[i]:endy[i]] = 0
        return masks

    @torch.no_grad()
    def inpaint(self, dataloader):
        '''
        Inpaint images
        :param dataloader: dataloader
        '''

        self.model.eval()
        for batch in tqdm(dataloader, desc='Inpainting', leave=True):
            # generate random rectangular masks with max size of half img_size
            batch = batch[0].to(self.device)
            original = batch.clone().cpu().numpy()
            if self.vae is not None:
                with torch.no_grad():
                    # if x has one channel, make it 3 channels
                    if batch.shape[1] == 1:
                        batch = torch.cat((batch, batch, batch), dim=1)
                    batch = self.encode(batch).latent_dist.sample().mul_(0.18215)

            masks = self.create_masks(batch.shape[0])
            inpainted_images = self.sampler.inpaint_loop(self.model, batch, masks, self.forward_diffusion_model)
            
            if self.vae is not None:
                with torch.no_grad():
                    inpainted_images = self.decode(inpainted_images.to(self.device) / 0.18215).sample 
            inpainted_images = inpainted_images.cpu().detach().numpy()

            break
                    
        plot_inpainting(original, masks, inpainted_images)
        


    
    @torch.no_grad()
    def sample(self, batch_size=16):
        '''
        Sample images
        :param batch_size: batch size
        '''
        samps = self.sampler.sample(model=self.model, image_size=self.img_size, batch_size=batch_size, channels=self.channels)[-1]
        if self.vae is not None:
            with torch.no_grad():
                samps = torch.tensor(samps, device=self.device)
                samps = self.decode(samps / 0.18215).sample 
                samps = samps.cpu().detach().numpy()
        plot_samples(samps)

    @torch.no_grad()
    def fid_sample(self, batch_size=16):
        '''
        Sample images for FID calculation
        :param batch_size: batch size
        '''

        # if self.args.checkpoint contains epoch number, ep = epoch number
        # else, ep = 0
        if 'epoch' in self.args.checkpoint:
            ep = int(self.args.checkpoint.split('epoch')[1].split('.')[0])
        else:
            ep = 0

        if not os.path.exists('./../../fid_samples'):
            os.makedirs('./../../fid_samples')
        if not os.path.exists(f"./../../fid_samples/{self.dataset}"):
            os.makedirs(f"./../../fid_samples/{self.dataset}")
        #add ddpm factor and timesteps
        if not os.path.exists(f"./../../fid_samples/{self.dataset}/ddpm_{self.args.ddpm}_timesteps_{self.args.sample_timesteps}_ep{ep}"):
            os.makedirs(f"./../../fid_samples/{self.dataset}/ddpm_{self.args.ddpm}_timesteps_{self.args.sample_timesteps}_ep{ep}")
        cnt = 0
        for i in tqdm(range(50000//batch_size), desc='FID Sampling', leave=True):
            samps = self.sampler.sample(model=self.model, image_size=self.img_size, batch_size=batch_size, channels=self.channels)[-1]

            if self.vae is not None:
                samps = torch.tensor(samps, device=self.device)
                samps = self.decode(samps / 0.18215).sample 
                samps = samps.cpu().detach().numpy()

            samps = samps * 0.5 + 0.5
            samps = samps.clip(0, 1)
            samps = samps.transpose(0,2,3,1)
            samps = (samps*255).astype(np.uint8)
            for samp in samps:
                cv2.imwrite(f"./../../fid_samples/{self.dataset}/ddpm_{self.args.ddpm}_timesteps_{self.args.sample_timesteps}_ep{ep}/{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1  

class LinearScheduler():
    def __init__(self, beta_start=0.0001, beta_end=0.02, timesteps=1000):
        '''
        Linear scheduler
        :param beta_start: starting beta value
        :param beta_end: ending beta value
        :param timesteps: number of timesteps
        '''
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._linear_beta_schedule()
        alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_one_by_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = self._compute_forward_diffusion_alphas(alphas_cumprod)
        self.posterior_variance = self._compute_posterior_variance(alphas_cumprod_prev, alphas_cumprod)

    def _compute_forward_diffusion_alphas(self, alphas_cumprod):
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    
    def _compute_posterior_variance(self, alphas_cumprod_prev, alphas_cumprod):
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        return self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def _linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)  

def extract_time_index(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
class ForwardDiffusion():
    def __init__(self, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, reverse_transform):
        '''
        Forward diffusion module
        :param sqrt_alphas_cumprod: square root of the cumulative product of alphas
        :param sqrt_one_minus_alphas_cumprod: square root of the cumulative product of 1 - alphas
        :param reverse_transform: reverse transform
        '''
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.reverse_transform = reverse_transform

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract_time_index(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_time_index(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noisy_image(self, x_start, t, noise=None):
        x_noisy = self.q_sample(x_start, t, noise)
        noisy_image = self.reverse_transform(x_noisy.squeeze())
        return noisy_image

class Sampler():
    def __init__(self, betas, timesteps=1000, sample_timesteps=100, ddpm=1.0, recon_factor=0.5):
        '''
        Sampler module
        :param betas: beta values
        :param timesteps: number of timesteps
        :param sample_timesteps: number of sample timesteps
        :param ddpm: diffusion coefficient
        :param recon_factor: starts reconstruction at recon_factor * timesteps
        '''
        self.betas = betas
        self.alphas = (1-self.betas).cumprod(dim=0)
        self.timesteps = timesteps
        self.sample_timesteps = sample_timesteps
        self.ddpm = ddpm
        self.scaling = timesteps//sample_timesteps
        self.recon_factor = recon_factor
    
    @torch.no_grad()
    def p_sample(self, model, x, t, tau_index):
        '''
        Sample from the model
        :param model: model
        :param x: input image
        :param t: time
        :param tau_index: tau index
        '''
        betas_t = extract_time_index(self.betas, t, x.shape)
        alpha_t = extract_time_index(self.alphas, t, x.shape)
        x0_t = (x - (1-alpha_t).sqrt()*model(x, t))/alpha_t.sqrt()

        if tau_index == (self.scaling-1): # last step
            return x0_t
        else:
            alpha_prev_t = extract_time_index(self.alphas, t-self.scaling, x.shape)
            c1 = self.ddpm*((1 - alpha_t/alpha_prev_t) * (1-alpha_prev_t) / (1 - alpha_t)).sqrt()
            c2  = ((1-alpha_prev_t) - c1**2).sqrt()
            noise = torch.randn_like(x)
            return x0_t*alpha_prev_t.sqrt() + c2*model(x,t) +  c1* noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        '''
        Sample from the model
        :param model: model
        :param shape: shape of the input image
        '''
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(range(self.timesteps-1,-1,-self.scaling), desc="Sampling", leave=False):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
        
        '''
        for i in tqdm(range(self.sample_timesteps-1,-1,-1), desc="Sampling", leave=False):
            scaled_i = i*self.scaling
            img = self.p_sample(model, img, torch.full((b,), scaled_i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
        '''
    
    @torch.no_grad
    def inpaint_loop(self, model, x, mask, forward):
        '''
        Inpaint the image
        :param model: model
        :param x: input image
        :param mask: mask
        '''
        device = next(model.parameters()).device

        b = x.shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn_like(x)
        noise = img.clone()
        img = mask * forward.q_sample(x, torch.full((b,), self.timesteps-1, device=device, dtype=torch.long), noise) + (1-mask) * img

        for i in tqdm(range(self.timesteps-1,-1,-self.scaling), desc="Sampling", leave=False):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            if i>0:
                x_i = forward.q_sample(x, torch.full((b,), max(0,i-self.scaling), device=device, dtype=torch.long), noise)
                img = mask * x_i + (1-mask) * img
            else:
                img = mask * x + (1-mask) * img

        return img

    @torch.no_grad
    def reconstruct_loop(self, model, shape, x):
        '''
        Reconstruct the image
        :param model: model
        :param shape: shape of the input image
        :param x: input image
        '''
        device = next(model.parameters()).device

        b = x.shape[0]
        # start from pure noise (for each example in the batch)
        img = x
        imgs = []
        initial_t = int(self.recon_factor*self.timesteps)
        
        for i in tqdm(range(initial_t-1,-1,-self.scaling), desc="Reconstructing", leave=False):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        '''
        Sample from the model
        :param model: model
        :param image_size: size of the image
        :param batch_size: batch size
        :param channels: number of channels
        '''
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def get_loss(forward_diffusion_model, denoising_model, x_start, t, noise=None, loss_type="l2"):
    '''
    Get the loss
    :param forward_diffusion_model: forward diffusion model
    :param denoising_model: denoising model
    :param x_start: input image
    :param t: time
    :param noise: noise
    :param loss_type: type of loss
    '''
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = forward_diffusion_model.q_sample(x_start=x_start, t=t, noise=noise).to(t.device)
    predicted_noise = denoising_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss