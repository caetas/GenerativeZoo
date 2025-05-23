import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import zuko
from torchvision.utils import make_grid
import torch.nn.functional as F
from functools import partial
from torch import einsum
from einops import rearrange
import math
import wandb
from config import models_dir
import os
from torchdiffeq import odeint
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
from collections import OrderedDict
import copy
from abc import abstractmethod
import cv2
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS

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
    :param n_classes: if specified (as an int), then this model will be
        class-conditional with `n_classes` classes.
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
        n_classes=None,
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
        self.n_classes = n_classes
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

        if self.n_classes is not None:
            self.label_emb = nn.Embedding(n_classes, time_embed_dim)

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
            self.n_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.n_classes is not None:
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
    '''
    Create a directory to save the model checkpoints
    '''
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, 'CondFlowMatching')):
        os.makedirs(os.path.join(models_dir, 'CondFlowMatching'))

class CondFlowMatching(nn.Module):

    def __init__(self, args, img_size=32, in_channels=3):
        '''
        FlowMatching module
        :param args: arguments
        :param img_size: size of the image
        :param in_channels: number of input channels
        '''
        super(CondFlowMatching, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(self.device) if args.latent else None
        self.channels = in_channels
        self.img_size = img_size

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
            n_classes=args.n_classes + 1,
            use_checkpoint=False,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=args.use_scale_shift_norm,
            resblock_updown=args.resblock_updown,
            use_new_attention_order=args.use_new_attention_order
        ).to(self.device)

        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.sample_and_save_freq = args.sample_and_save_freq
        self.dataset = args.dataset
        self.solver = args.solver
        self.step_size = args.step_size
        self.solver_lib = args.solver_lib
        self.n_classes = args.n_classes
        self.dropout_prob = args.dropout_prob
        self.cfg = args.cfg
        self.no_wandb = args.no_wandb
        self.warmup = args.warmup
        self.decay = args.decay
        self.num_samples = args.num_samples
        self.snapshot = args.n_epochs // args.snapshots
        self.translation_factor = args.translation_factor
        self.recon_factor = args.recon_factor
        self.lpips = args.lpips
        if args.train:
            self.ema = copy.deepcopy(self.model)
            self.ema_rate = args.ema_rate
            for param in self.ema.parameters():
                param.requires_grad = False

    def forward(self, x, t, c):
        '''
        Forward pass of the FlowMatching module
        :param x: input image
        :param t: time
        '''
        return self.model(x, t, c)
    
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
    
    
    def conditional_flow_matching_loss(self, x, c):
        '''
        Conditional flow matching loss
        :param x: input image
        '''
        sigma_min = 1e-4
        t = torch.rand(x.shape[0], device=x.device)
        # select self.dropout_prob*x.shape[0] random indices
        indices = torch.randperm(x.shape[0])[:int(self.dropout_prob*x.shape[0])]
        c[indices] = self.n_classes
        noise = torch.randn_like(x)

        x_t = (1 - (1 - sigma_min) * t[:, None, None, None]) * noise + t[:, None, None, None] * x
        optimal_flow = x - (1 - sigma_min) * noise
        predicted_flow = self.forward(x_t, t, c)

        return (predicted_flow - optimal_flow).square().mean()
    
    @torch.no_grad()
    def sample(self, n_samples, train=True, accelerate=None, label=None, fid=False, x_0=None, start=0):
        '''
        Sample images
        :param n_samples: number of samples
        :param train: if True, log the samples to wandb
        '''
        if x_0 is None:
            x_0 = torch.randn(n_samples, self.channels, self.img_size, self.img_size, device=self.device)
        if label is None:
            y = torch.arange(n_samples, device=self.device).long() % self.n_classes
        else:
            y = label
        # concatenate y with a tensor like y but with all elements equal to self.n_classes
        y = torch.cat([y, self.n_classes*torch.ones(n_samples, device=self.device).long()])

        if train:
            def f(t: float, x):
                x = x.repeat(2,1,1,1)
                v = self.ema(x, torch.full(x.shape[:1], t, device=self.device), y.to(self.device))
                vc = v[:n_samples]
                vu = v[n_samples:]
                return vu + (vc - vu)*self.cfg
        else:
            def f(t: float, x):
                x = x.repeat(2,1,1,1)
                v = self.forward(x, torch.full(x.shape[:1], t, device=self.device), y.to(self.device))
                vc = v[:n_samples]
                vu = v[n_samples:]
                return vu + (vc - vu)*self.cfg
        
        if self.solver_lib == 'torchdiffeq':
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_0, t=torch.linspace(start, 1, 2).to(self.device), options={'step_size': self.step_size}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_0, t=torch.linspace(start, 1, 2).to(self.device), method=self.solver, options={'max_num_steps': 1//self.step_size}, rtol=1e-5, atol=1e-5)
            samples = samples[1]
        elif self.solver_lib == 'zuko':
            samples = zuko.utils.odeint(f, x_0, 0, 1, phi=self.model.parameters(), atol=1e-5, rtol=1e-5)
        else:
            t = 0.
            for i in tqdm(range(int(1/self.step_size)), desc='Sampling', leave=False):
                x_0 = x_0.repeat(2,1,1,1)
                if train:
                    v = self.ema(x_0, torch.full(x_0.shape[:1], t, device=self.device), y.to(self.device))
                else:
                    v = self.forward(x_0, torch.full(x_0.shape[:1], t, device=self.device), y.to(self.device))
                vc = v[:n_samples]
                vu = v[n_samples:]
                v = vu + (vc - vu)*self.cfg
                x_0 = x_0[:n_samples] + self.step_size * v
                t += self.step_size
            samples = x_0

        if self.vae is not None:
            samples = self.decode(samples / 0.18215).sample
        samples = samples*0.5 + 0.5
        samples = samples.clamp(0, 1)
        if fid:
            return samples
        fig = plt.figure(figsize=(10, 10))
        grid = make_grid(samples, nrow=int(np.sqrt(n_samples)), padding=0)
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')

        if train:
            if not self.no_wandb:
                accelerate.log({"samples": fig})
        else:
            plt.show()

        plt.close(fig)

    @torch.no_grad()
    def latent(self, x_1, label, end=0):
        '''
        Get the x_0 from the input image
        :param label: label of the input image
        :param x_1: input image
        '''
        if self.vae is not None:
            if x_1.shape[1] == 1:
                x_1 = x_1.repeat(1, 3, 1, 1)
            x_1 = self.encode(x_1).latent_dist.sample().mul_(0.18215)

        label = torch.cat([label, self.n_classes*torch.ones(x_1.shape[0], device=self.device).long()])
        
        def f(t: float, x):
            x = x.repeat(2,1,1,1)
            v = self.model(x, torch.full(x.shape[:1], t, device=self.device), label.to(self.device))
            vc = v[:x.shape[0]//2]
            vu = v[x.shape[0]//2:]
            return vu + (vc - vu)*self.cfg
        
        if self.solver_lib == 'torchdiffeq':
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_1, t=torch.linspace(1, end, 2).to(self.device), options={'step_size': self.step_size}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_1, t=torch.linspace(1, end, 2).to(self.device), method=self.solver, options={'max_num_steps': 1//self.step_size}, rtol=1e-5, atol=1e-5)
            samples = samples[-1]

        elif self.solver_lib == 'zuko':
            samples = zuko.utils.odeint(f, x_1, 1, 0, phi=self.model.parameters(), atol=1e-5, rtol=1e-5)

        else:
            t = 1.
            for i in tqdm(range(int(1/self.step_size)), desc='Latent', leave=False):
                x_1 = x_1.repeat(2,1,1,1)
                v = self.model(x_1, torch.full(x_1.shape[:1], t, device=self.device), label.to(self.device))
                vc = v[:x_1.shape[0]//2]
                vu = v[x_1.shape[0]//2:]
                v = vu + (vc - vu)*self.cfg
                x_1 = x_1[:x_1.shape[0]//2] - self.step_size * v
                t -= self.step_size

            samples = x_1
        return samples
    
    @torch.no_grad()
    def image_translation(self, val_loader):
        '''
        Image translation
        :param x_1: input image
        :param label: label of the input image
        '''
        # only one batch
        x_1, label = next(iter(val_loader))
        # if label has an extra dimension, squeeze it
        if label.dim() > 1:
            label = label.squeeze(1)

        x_1 = x_1.to(self.device)
        label = label.to(self.device)
        x_0 = self.latent(x_1, label, end=self.translation_factor)
        label = (label + 1).long() % self.n_classes
        x_1_translated = self.sample(x_1.shape[0], train=False, label=label, x_0=x_0, fid=True, start=self.translation_factor)

        #plot x_1 and x_1_translated side by side, make grids
        x_1 = x_1*0.5 + 0.5
        x_1 = x_1.clamp(0, 1)
        grid1 = make_grid(x_1, nrow=int(np.sqrt(x_1.shape[0])), padding=0)
        grid2 = make_grid(x_1_translated, nrow=int(np.sqrt(x_1_translated.shape[0])), padding=0)
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(grid1.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(grid2.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        plt.show()

    @torch.no_grad()
    def classification(self, val_loader):
        '''
        Classification
        :param val_loader: validation data loader
        '''
        self.model.eval()

        gt = []
        pred = []
        pred_recon = []

        pred_all = []
        pred_recon_all = []

        if self.lpips:
            lpips_metric = LPIPS(net='vgg').to(self.device)
        # two modes: encoding and reconstruction
        for x_1, label in tqdm(val_loader, desc='Classification', leave=False):
            x_1 = x_1.to(self.device)

            if label.dim() > 1:
                label = label.squeeze(1)
            label = label.to(self.device)
            #aux = self.cfg
            #self.cfg = 0.0
            #x_0 = self.latent(x_1.to(self.device), label.to(self.device), end=self.translation_factor)
            #self.cfg = aux
            error = torch.zeros((x_1.shape[0], self.n_classes), device=self.device)
            lpips_error = torch.zeros((x_1.shape[0], self.n_classes), device=self.device)
            error_recon = torch.zeros((x_1.shape[0], self.n_classes), device=self.device)
            lpips_error_recon = torch.zeros((x_1.shape[0], self.n_classes), device=self.device)

            for i in range(self.n_classes):
                cl = i*torch.ones(x_1.shape[0], device=self.device).long()

                ### Translation
                aux = self.cfg
                self.cfg = 5.0
                x_0 = self.latent(x_1, cl, end=self.translation_factor)
                xnorm = torch.linalg.norm(x_0, dim=1, keepdim=True, ord=2).flatten(1)
                for j in range(xnorm.shape[0]):
                    error[j,i] = torch.abs(xnorm[j].max() - xnorm[j].min())
                    # add euclidean norm of x_0 to error
                    error[j,i] += torch.linalg.norm(x_0[j], dim=0, ord=2).mean() 
                
                self.cfg = aux
                #x_1_translated = self.sample(x_1.shape[0], train=False, label=cl, x_0=x_0, fid=True, start=self.translation_factor)
                #error[:, i] = ((x_1*0.5 +0.5).clamp(0,1) - x_1_translated).square().mean(dim=(1, 2, 3))
                if self.lpips:
                    if x_1.shape[1] == 1:
                        x_1 = x_1.repeat(1, 3, 1, 1)
                    if x_1_translated.shape[1] == 1:
                        x_1_translated = x_1_translated.repeat(1, 3, 1, 1)
                    lpips_error[:, i] = lpips_metric(x_1.clamp(-1,1), (x_1_translated*2 -1).clamp(-1,1)).view(-1)
                # plot x_1 and x_1_translated side by side, make grids         
                '''
                grid1 = make_grid((x_1*0.5 + 0.5).clamp(0,1), nrow=int(np.sqrt(x_1.shape[0])), padding=0)
                grid2 = make_grid(x_1_translated.clamp(0, 1), nrow=int(np.sqrt(x_1_translated.shape[0])), padding=0)
                fig = plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(grid1.permute(1, 2, 0).cpu().detach().numpy())
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(grid2.permute(1, 2, 0).cpu().detach().numpy())
                plt.axis('off')
                plt.show()
                '''
                
                # Reconstruction
                for j in range(1):
                    if self.vae is not None:
                        if x_1.shape[1] == 1:
                            x_1 = x_1.repeat(1, 3, 1, 1)
                        x_1_encode = self.encode(x_1).latent_dist.sample().mul_(0.18215)
                        noise = torch.randn_like(x_1_encode)
                        x_t = (1 - (1 - 1e-7) * self.recon_factor) * noise + self.recon_factor * x_1_encode
                    else:
                        x_t = (1 - (1 - 1e-7) * self.recon_factor) * torch.randn_like(x_1) + self.recon_factor * x_1
                    x_1_recon = self.sample(x_1.shape[0], train=False, label=cl, x_0=x_t, fid=True, start=self.recon_factor)
                    error_recon[:, i] = ((x_1*0.5 + 0.5).clamp(0,1) - x_1_recon).square().mean(dim=(1, 2, 3))
                    if self.lpips:
                        if x_1.shape[1] == 1:
                            x_1 = x_1.repeat(1, 3, 1, 1)
                        if x_1_recon.shape[1] == 1:
                            x_1_recon = x_1_recon.repeat(1, 3, 1, 1)
                        lpips_error_recon[:, i] = lpips_metric(x_1.clamp(-1,1), (x_1_recon*2 -1).clamp(-1,1)).view(-1)

            error_recon = error_recon/torch.sum(error_recon, dim=1, keepdims=True) + ((lpips_error_recon/torch.sum(lpips_error_recon, dim=1, keepdims=True)) if self.lpips else 0)
            error = error/torch.sum(error, dim=1, keepdims=True) + ((lpips_error/torch.sum(lpips_error, dim=1, keepdims=True)) if self.lpips else 0)

            # get the index of the minimum error for Accuracy
            pred.append(torch.argmin(error, dim=1).cpu().numpy())
            pred_recon.append(torch.argmin(error_recon, dim=1).cpu().numpy())
            pred_all.append(error.cpu().numpy())
            pred_recon_all.append(error_recon.cpu().numpy())
            gt.append(label.cpu().numpy())


        gt = np.concatenate(gt)
        pred = np.concatenate(pred)
        pred_recon = np.concatenate(pred_recon)
        pred_all = np.concatenate(pred_all)
        pred_recon_all = np.concatenate(pred_recon_all)

        # get the accuracy
        acc = np.sum(gt == pred) / len(gt)
        acc_recon = np.sum(gt == pred_recon) / len(gt)

        # get auc for each class
        auc = np.zeros(self.n_classes)
        auc_recon = np.zeros(self.n_classes)

        for i in range(self.n_classes):
            auc[i] = roc_auc_score(gt != i, pred_all[:, i])
            auc_recon[i] = roc_auc_score(gt != i, pred_recon_all[:, i])
        auc = np.mean(auc)
        auc_recon = np.mean(auc_recon)

        print(f'Accuracy Translation: {acc*100:.2f}%')
        print(f'Accuracy Reconstruction: {acc_recon*100:.2f}%')
        print(f'AUC Translation: {auc*100:.2f}%')
        print(f'AUC Reconstruction: {auc_recon*100:.2f}%')


    
    def train_model(self, train_loader, verbose=True):
        '''
        Train the model
        :param train_loader: training data loader
        '''
        accelerate = Accelerator(log_with="wandb")
        if not self.no_wandb:
            accelerate.init_trackers(project_name='CondFlowMatching',
            config = {
                        "dataset": self.args.dataset,
                        "batch_size": self.args.batch_size,
                        "n_epochs": self.args.n_epochs,
                        "lr": self.args.lr,
                        "channels": self.channels,
                        "input_size": self.img_size,
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
                        "dropout_prob": self.args.dropout_prob,
                        "cfg": self.args.cfg,   
                },
                init_kwargs={"wandb":{"name": f"CondFlowMatching_{self.args.dataset}"}})
            

        epoch_bar = tqdm(range(self.n_epochs), desc='Epochs', leave=True)
        create_checkpoint_dir()

        best_loss = float('inf')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.n_epochs*len(train_loader), pct_start=self.warmup/self.n_epochs, anneal_strategy='cos', cycle_momentum=False, div_factor=self.lr/1e-6, final_div_factor=1)

        if  self.vae is None:
            train_loader, self.model, optimizer, scheduler, self.ema = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema)
        else:
            train_loader, self.model, optimizer, scheduler, self.ema, self.vae = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema, self.vae)

        update_ema(self.ema, self.model, 0)

        for epoch in epoch_bar:
            self.model.train()
            train_loss = 0.0
            for x, cl in tqdm(train_loader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)
                # if cl has an extra dimension, squeeze it
                if cl.dim() > 1:
                    cl = cl.squeeze(1)

                with accelerate.autocast():

                    if self.vae is not None:
                        with torch.no_grad():
                            # if x has one channel, make it 3 channels
                            if x.shape[1] == 1:
                                x = torch.cat((x, x, x), dim=1)
                            x = self.encode(x).latent_dist.sample().mul_(0.18215)

                    cl = cl.to(x.device)
                    optimizer.zero_grad()
                    loss = self.conditional_flow_matching_loss(x, cl)
                    accelerate.backward(loss)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()*x.size(0)
                update_ema(self.ema, self.model, self.ema_rate)

            accelerate.wait_for_everyone()

            if not self.no_wandb:
                accelerate.log({"Train Loss": train_loss / len(train_loader.dataset)})
                accelerate.log({"Learning Rate": scheduler.get_last_lr()[0]})

            epoch_bar.set_postfix({'Loss': train_loss / len(train_loader.dataset)})

            if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:
                self.model.eval()
                self.sample(self.num_samples, train=True, accelerate=accelerate)
            
            if train_loss < best_loss:
                best_loss = train_loss
            
            if (epoch+1) % self.snapshot == 0:
                ema_to_save = accelerate.unwrap_model(self.ema)
                accelerate.save(ema_to_save.state_dict(), os.path.join(models_dir, 'CondFlowMatching', f"{'LatCondFM' if self.vae is not None else 'CondFM'}_{self.dataset}_epoch{epoch+1}.pt"))

        accelerate.end_training()

    def load_checkpoint(self, checkpoint_path):
        '''
        Load a model checkpoint
        :param checkpoint_path: path to the checkpoint
        '''
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    
    @torch.no_grad()
    def fid_sample(self):

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
        if not os.path.exists(f"./../../fid_samples/{self.dataset}/condfm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}_w{self.cfg}"):
            os.makedirs(f"./../../fid_samples/{self.dataset}/condfm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}_w{self.cfg}")
        cnt = 0

        self.model.eval()
        samp_per_class = 50000//self.n_classes
        its_per_class = samp_per_class//self.args.batch_size

        for c in tqdm(range(self.n_classes), desc="Class", leave=True):
            for i in tqdm(range(its_per_class), desc="Iteration", leave=False):
                samples = self.sample(self.args.batch_size, train=False, label=torch.full((self.args.batch_size,), c, device=self.device).long(), fid=True)
                samples = samples.permute(0,2,3,1).cpu().numpy()
                samples = (samples*255).astype(np.uint8)
                for samp in samples:
                    cv2.imwrite(f"./../../fid_samples/{self.dataset}/condfm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}_w{self.cfg}/{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                    cnt += 1 
