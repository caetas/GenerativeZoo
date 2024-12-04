###################################################################################################################
####### Code based on https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing #######
###################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from tqdm import trange, tqdm
from config import models_dir
import os
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score
import wandb
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
from collections import OrderedDict
import copy
from abc import abstractmethod
import math

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

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'SGM')):
    os.makedirs(os.path.join(models_dir, 'SGM'))

class SGM(nn.Module):
    def __init__(self, args, in_channels=1, input_size=32):
        '''
        Vanilla SGM model
        Args:
            args: The arguments for the model
            in_channels: The number of input channels.
            input_size: The size of the input image.
        '''
        super(SGM, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigma = args.sigma
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.in_channels = in_channels
        self.input_size = input_size

        self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(self.device) if args.latent else None
        self.channels = in_channels
        self.img_size = input_size

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
                num_classes= args.n_classes + 1 if args.conditional else None,
                use_checkpoint=False,
                num_heads=args.num_heads,
                num_head_channels=args.num_head_channels,
                num_heads_upsample=-1,
                use_scale_shift_norm=args.use_scale_shift_norm,
                resblock_updown=args.resblock_updown,
                use_new_attention_order=args.use_new_attention_order
            ).to(self.device)

        self.dataset = args.dataset
        self.sample_and_save_freq = args.sample_and_save_freq
        self.num_steps = args.num_steps
        self.solver = args.solver
        self.atol = args.atol
        self.rtol = args.rtol
        self.eps = args.eps
        self.no_wandb = args.no_wandb
        self.decay = args.decay
        self.warmup = args.warmup
        self.latent = args.latent
        self.conditional = args.conditional
        self.n_classes = args.n_classes
        self.cfg = args.cfg
        self.drop_prob = args.drop_prob
        self.sampler = args.sampler
        self.snr = args.snr

        if args.train:
                self.ema = copy.deepcopy(self.model)
                self.ema_rate = args.ema_rate
                for param in self.ema.parameters():
                    param.requires_grad = False

    def train_model(self, dataloader, verbose=True):
        '''
        Train the Vanilla SGM model
        Args:
            dataloader: A PyTorch dataloader that provides training data.
        '''

        accelerate = Accelerator(log_with="wandb")
        if not self.no_wandb:
            accelerate.init_trackers(project_name='SGM',
            config = {
                        'dataset': self.args.dataset,
                        'batch_size': self.args.batch_size,
                        'n_epochs': self.args.n_epochs,
                        'lr': self.args.lr,
                        'sigma': self.args.sigma,
                        'input_size': self.img_size,
                        'channels': self.channels,
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
                        "cfg": self.args.cfg,
                        "drop_prob": self.args.drop_prob,
                        "n_classes": self.n_classes,  
                },
                init_kwargs={"wandb":{"name": f"SGM_{self.args.dataset}"}})

        epoch_bar = trange(self.n_epochs, desc='Average loss: N/A')
        best_loss = np.inf

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.n_epochs, pct_start=self.warmup/self.n_epochs, anneal_strategy='cos', cycle_momentum=False, div_factor=self.lr/1e-6, final_div_factor=1)

        create_checkpoint_dir()

        update_ema(self.ema, self.model, 0)

        dataloader, self.model, optimizer, scheduler = accelerate.prepare(dataloader, self.model, optimizer, scheduler)

        for epoch in epoch_bar:
            avg_loss = 0.0
            
            for x,y in tqdm(dataloader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)
                y = y.to(self.device)

                if self.vae is not None:
                    with torch.no_grad():
                        # if x has one channel, make it 3 channels
                        if x.shape[1] == 1:
                            x = torch.cat((x, x, x), dim=1)
                        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)

                optimizer.zero_grad()
                loss = self.loss_fn(x, y)

                accelerate.backward(loss)
                optimizer.step()

                avg_loss += loss.item()*x.shape[0]
                update_ema(self.ema, self.model, self.ema_rate)

            epoch_bar.set_description('Average Loss: {:5f}'.format(avg_loss / len(dataloader.dataset)))
            if not self.no_wandb:
                accelerate.log({'loss': avg_loss / len(dataloader.dataset)}, step = epoch)
                accelerate.log({'lr': scheduler.get_last_lr()[0]}, step = epoch)
            
            scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.ema.state_dict(), os.path.join(models_dir,'SGM', f"{'Cond' if self.conditional else ''}{'LatSGM' if self.latent else 'SGM'}_{self.dataset}.pt"))

            if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:

                samples = self.get_samples(16)
                if self.vae is not None:
                    samples = self.vae.decode(samples.to(self.device) / 0.18215).sample 
                samples = samples*0.5 + 0.5
                samples = samples.clamp(0.0, 1.0)
                sample_grid = make_grid(samples, nrow=int(np.sqrt(16)), padding=0)

                fig = plt.figure(figsize=(10, 10))
                plt.imshow(sample_grid.permute(1, 2, 0).cpu().numpy(), vmin=0.0, vmax=1.0)
                plt.axis('off')
                if not self.no_wandb:
                    accelerate.log({'samples': fig}, step = epoch)
                plt.close(fig)

    def marginal_prob_std(self, t):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:    
            t: A vector of time steps.
            sigma: The $\sigma$ in our SDE.  

        Returns:
            The standard deviation.
        """    
        #t = torch.tensor(t, device=device)
        t = t.clone().detach().to(self.device)
        return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))

    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient of our SDE.

        Args:
            t: A vector of time steps.
            sigma: The $\sigma$ in our SDE.

        Returns:
            The vector of diffusion coefficients.
        """
        return (self.sigma**t).to(self.device)

    def loss_fn(self, x, y=None):
        """The loss function for training score-based generative models.

        Args:
            model: A PyTorch model instance that represents a 
            time-dependent score-based model.
            x: A mini-batch of training data.    
            marginal_prob_std: A function that gives the standard deviation of 
            the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - self.eps) + self.eps  
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        if self.conditional:
            y = torch.where(torch.rand(x.shape[0], device=self.device) < self.drop_prob, torch.full((x.shape[0],), self.n_classes, device=self.device), y)
            score = self.model(perturbed_x, random_t, y)
        else:
            score = self.model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss
  
    @torch.no_grad()
    def ode_sampler(self, num_samples, train):
        """Generate samples from score-based models with black-box ODE solvers.

        Args:
        num_samples: The number of samples to generate.
        train: A boolean that indicates whether to use the training or the
        """
        t = torch.ones(num_samples, device=self.device)
        # Create the latent code
        init_x = torch.randn(num_samples, self.channels, self.img_size, self.img_size, device=self.device)* self.marginal_prob_std(t)[:, None, None, None]

        shape = init_x.shape
        y = torch.arange(num_samples, device=self.device)%self.n_classes if self.conditional else None

        def score_eval_wrapper(sample, time_steps):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = sample.float().to(self.device).reshape(shape) #torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(shape).clone().detach()
            time_steps = time_steps.float().to(self.device).reshape((sample.shape[0], )) #torch.tensor(time_steps, device=self.device, dtype=torch.float32).reshape((sample.shape[0], )).clone().detach()
            if self.conditional:
                with torch.no_grad():
                    score_cond = self.model(sample, time_steps, y) if not(train) else self.ema(sample, time_steps, y)
                    score_uncond = self.model(sample, time_steps, torch.full((sample.shape[0],), self.n_classes, device=self.device)) if not(train) else self.ema(sample, time_steps, torch.full((sample.shape[0],), self.n_classes, device=self.device))
                    score = self.cfg*score_cond + (1-self.cfg)*score_uncond
            else:    
                with torch.no_grad():    
                    score = self.model(sample, time_steps) if not(train) else self.ema(sample, time_steps)
            return score

        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = torch.ones((shape[0],), device=t.device) * t    
            g = self.diffusion_coeff(t)
            return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

        # Run the black-box ODE solver.
        if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
            samples = odeint(ode_func, init_x, t=torch.linspace(1, self.eps, 2).to(self.device), options={'step_size': 1./self.num_steps}, method=self.solver, rtol=self.rtol, atol=self.atol)
        else:
            samples = odeint(ode_func, init_x, t=torch.linspace(1, self.eps, 2).to(self.device), method=self.solver, options={'max_num_steps': self.num_steps}, rtol=self.rtol, atol=self.atol)

        x = samples[1]

        return x
    
    @torch.no_grad()
    def em_sampler(self, num_samples, train):
        """Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
        num_samples: The number of samples to generate.
        train: A boolean that indicates whether to use the training or the test model.  
        """
        t = torch.ones(num_samples, device=self.device)
        init_x = torch.randn(num_samples, self.channels, self.img_size, self.img_size, device=self.device)* self.marginal_prob_std(t)[:, None, None, None]
        time_steps = torch.linspace(1., self.eps, self.num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        y = torch.arange(num_samples, device=self.device)%self.n_classes if self.conditional else None
        for time_step in tqdm(time_steps):     
            batch_time_step = torch.ones(num_samples, device=self.device) * time_step 
            g = self.diffusion_coeff(batch_time_step)
            if self.conditional:
                score_cond = self.model(x, batch_time_step, y) if not(train) else self.ema(x, batch_time_step, y)
                score_uncond = self.model(x, batch_time_step, torch.full((x.shape[0],), self.n_classes, device=self.device)) if not(train) else self.ema(x, batch_time_step, torch.full((x.shape[0],), self.n_classes, device=self.device))
                score = self.cfg*score_cond + (1-self.cfg)*score_uncond
            else:
                score = self.model(x, batch_time_step) if not(train) else self.ema(x, batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
        # Do not include any noise in the last sampling step.
        return mean_x
    
    @torch.no_grad()
    def pc_sampler(self, num_samples, train):

        """Generate samples from score-based models with Predictor-Corrector method.

        Args:
        num_samples: The number of samples to generate.
        train: A boolean that indicates whether to use the training or the test model.  
        """
        t = torch.ones(num_samples, device=self.device)
        init_x = torch.randn(num_samples, self.channels, self.img_size, self.img_size, device=self.device)* self.marginal_prob_std(t)[:, None, None, None]
        time_steps = torch.linspace(1., self.eps, self.num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        y = torch.arange(num_samples, device=self.device)%self.n_classes if self.conditional else None

        for time_step in tqdm(time_steps, desc = 'PC sampling', leave=False):      
            batch_time_step = torch.ones(num_samples, device=self.device) * time_step
            # Corrector step (Langevin MCMC)
            if self.conditional:
                grad_cond = self.model(x, batch_time_step, y) if not(train) else self.ema(x, batch_time_step, y)
                grad_uncond = self.model(x, batch_time_step, torch.full((x.shape[0],), self.n_classes, device=self.device)) if not(train) else self.ema(x, batch_time_step, torch.full((x.shape[0],), self.n_classes, device=self.device))
                grad = self.cfg*grad_cond + (1-self.cfg)*grad_uncond
            else:
                grad = self.model(x, batch_time_step) if not(train) else self.ema(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (self.snr * noise_norm / grad_norm)**2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

            # Predictor step (Euler-Maruyama)
            g = self.diffusion_coeff(batch_time_step)
            if self.conditional:
                score_cond = self.model(x, batch_time_step, y) if not(train) else self.ema(x, batch_time_step, y)
                score_uncond = self.model(x, batch_time_step, torch.full((x.shape[0],), self.n_classes, device=self.device)) if not(train) else self.ema(x, batch_time_step, torch.full((x.shape[0],), self.n_classes, device=self.device))
                score = self.cfg*score_cond + (1-self.cfg)*score_uncond
            else:
                score = self.model(x, batch_time_step) if not(train) else self.ema(x, batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score * step_size    
            
        # The last step does not include any noise
        return mean_x
        
    
    @torch.no_grad()
    def get_samples(self, num_samples=16, train=False):
        """Generate samples from score-based models with black-box ODE solvers.

        Args:
        num_samples: The number of samples to generate.
        train: A boolean that indicates whether to use the training or the
        """
        if self.sampler == 'ode':
            return self.ode_sampler(num_samples, train)
        elif self.sampler == 'pc':
            return self.pc_sampler(num_samples, train)
        elif self.sampler == 'em':
            return self.em_sampler(num_samples, train)
        else:
            raise ValueError(f"Invalid sampler type: {self.sampler}, pick one of 'ode', 'pc', or 'em'.")
  
    @torch.no_grad()
    def sample(self, num_samples=64):
        '''
        Sample from the Vanilla SGM model
        Args:
            num_samples: The number of samples to generate.
            num_steps: The number of sampling steps.
            snr: The signal-to-noise ratio for the PC sampler.
            sampler_type: The type of sampler. One of 'EM', 'PC', and 'ODE'.
        '''
        self.model.eval()
        samples = self.get_samples(num_samples)
        if self.vae is not None:
            samples = self.vae.decode(samples.to(self.device) / 0.18215).sample 
        samples = samples*0.5 + 0.5  
        samples = samples.clamp(0.0, 1.0)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(num_samples)), padding=0)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(sample_grid.permute(1, 2, 0).cpu().numpy(), vmin=0.0, vmax=1.0)
        plt.axis('off')
        plt.show()


'''
def Euler_Maruyama_sampler(self, num_samples=64):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.    
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(num_samples, self.channels, self.img_size, self.img_size, device=self.device)* self.marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., self.eps, self.num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
    for time_step in tqdm(time_steps):     
        batch_time_step = torch.ones(num_samples, device=self.device) * time_step 
        g = self.diffusion_coeff(batch_time_step)
        mean_x = x + (g**2)[:, None, None, None] * self.model(x, batch_time_step) * step_size
        x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
    # Do not include any noise in the last sampling step.
    return mean_x

def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=500, 
               snr=0.16,                
               device='cuda',
               eps=1e-3,
               channels=1,
               input_size=32):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(num_samples, device=self.device)
  init_x = torch.randn(bat, channels, input_size, input_size, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps, desc = 'PC sampling', leave=False):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean
'''