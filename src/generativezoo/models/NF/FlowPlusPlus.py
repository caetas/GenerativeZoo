#################################################################
### Code based on: https://github.com/chrischute/flowplusplus ###
#################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import wandb
from config import models_dir
import os

def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(models_dir + '/FlowPP'):
        os.makedirs(models_dir + '/FlowPP')

class FlowPlusPlus(nn.Module):
    """Flow++ Model

    Based on the paper:
    "Flow++: Improving Flow-Based Generative Models
        with Variational Dequantization and Architecture Design"
    by Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, Pieter Abbeel
    (https://openreview.net/forum?id=Hyg74h05tX).

    Args:
        scales (tuple or list): Number of each type of coupling layer in each
            scale. Each scale is a 2-tuple of the form
            (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_dequant_blocks (int): Number of blocks in the dequantization flows.
    """
    def __init__(self, args,
                 scales=((0, 4), (2, 3)),
                 channels = 3,
                 img_size = 32):
        super(FlowPlusPlus, self).__init__()
        # Register bounds to pre-process images, not learnable
        in_shape = (channels, img_size, img_size)
        self.mid_channels = args.num_channels
        self.num_blocks = args.num_blocks
        self.num_components = args.num_components
        self.use_attn = args.use_attn
        self.drop_prob = args.drop_prob
        self.num_dequant_blocks = args.num_dequant_blocks

        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        if self.num_dequant_blocks > 0:
            self.dequant_flows = _Dequantization(in_shape=in_shape,
                                                 mid_channels=self.mid_channels,
                                                 num_blocks=self.num_dequant_blocks,
                                                 use_attn=self.use_attn,
                                                 drop_prob=self.drop_prob)
        else:
            self.dequant_flows = None
        self.flows = _FlowStep(scales=scales,
                               in_shape=in_shape,
                               mid_channels=self.mid_channels,
                               num_blocks=self.num_blocks,
                               num_components=self.num_components,
                               use_attn=self.use_attn,
                               drop_prob=self.drop_prob)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channels = channels
        self.img_size = img_size
        self.flows.to(self.device)
        self.dequant_flows.to(self.device)
        self.no_wandb = args.no_wandb

    def forward(self, x, reverse=False):
        sldj = torch.zeros(x.size(0), device=x.device)
        if not reverse:
            x, sldj = self.dequantize(x, sldj)
            x, sldj = self.to_logits(x, sldj)
        x, sldj = self.flows(x, sldj, reverse)

        return x, sldj

    def dequantize(self, x, sldj):
        if self.dequant_flows is not None:
            x, sldj = self.dequant_flows(x, sldj)
        else:
            x = (x * 255. + torch.rand_like(x)) / 256.

        return x, sldj

    def to_logits(self, x, sldj):
        """Convert the input image `x` to logits.

        Args:
            x (torch.Tensor): Input image.
            sldj (torch.Tensor): Sum log-determinant of Jacobian.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        y = (2 * x - 1) * self.bounds.to(x.device)
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.bounds.to(x.device)).log() - self.bounds.to(x.device).log())
        sldj = sldj + ldj.flatten(1).sum(-1)

        return y, sldj
    
    def load_checkpoints(self, args):
        if args.checkpoint is not None:
            self.flows.load_state_dict(torch.load(args.checkpoint))
            self.dequant_flows.load_state_dict(torch.load(args.checkpoint.replace('FlowPP_', 'DequantFlowPP_')))
    
    @torch.enable_grad()
    def train_model(self, args, train_loader, verbose=True):
        """Train a Flow++ model.

        Args:
            args (argparse.Namespace): Command-line arguments.
            train_loader (DataLoader): Training data loader.

        """
        create_checkpoint_dir()

        global global_step
        global_step = 0

        loss_fn = NLLLoss().to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

        warm_up = args.warm_up*args.batch_size
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., step / warm_up))

        tbar = trange(args.n_epochs, desc='Training')
        best_loss = np.inf

        for epoch in tbar:
            self.train()
            total_loss = 0.
            for (x, _) in tqdm(train_loader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)

                optimizer.zero_grad()
                z, sldj = self.forward(x)
                loss = loss_fn(z, sldj)
                loss.backward()

                if args.grad_clip > 0:
                    clip_grad_norm(optimizer, args.grad_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()*x.size(0)
                global_step += x.size(0)

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.flows.state_dict(), os.path.join(models_dir, 'FlowPP', f'FlowPP_{args.dataset}.pt'))
                torch.save(self.dequant_flows.state_dict(), os.path.join(models_dir, 'FlowPP', f'DequantFlowPP_{args.dataset}.pt'))

            tbar.set_postfix(loss=total_loss/len(train_loader))
            if not self.no_wandb:
                wandb.log({'train_loss': total_loss/len(train_loader)})

            if (epoch+1) % args.sample_and_save_freq == 0 or epoch == 0:
                self.sample(16)

    @torch.no_grad()
    def sample(self, num_samples, train=True):
        """Sample from a Flow++ model.

        Args:
            num_samples (int): Number of samples to generate.
        """
        self.eval()
        samples = torch.randn(num_samples, self.channels, self.img_size, self.img_size, device=self.device)
        samples, _ = self.forward(samples, reverse=True)
        samples = torch.sigmoid(samples)

        # Plot samples
        samples = make_grid(samples, nrow=int(num_samples ** 0.5), padding=0)
        samples = samples.permute(1, 2, 0).cpu().numpy()
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(samples)
        plt.axis('off')
        if train:
            if not self.no_wandb:
                wandb.log({'samples': fig})
        else:
            plt.show()
        plt.close(fig)

    def nll_scores(self, z, sldj):
        """Compute negative log-likelihood scores.

        Args:
            z (torch.Tensor): Latent representation.
            sldj (torch.Tensor): Sum log-determinant of Jacobian.

        Returns:
            scores (torch.Tensor): Negative log-likelihood scores.
        """
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(256) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        return -ll

    def outlier_detection(self, in_loader, out_loader):
        """Outlier detection using a Flow++ model.

        Args:
            in_loader (DataLoader): In-distribution data loader.
            out_loader (DataLoader): Out-of-distribution data loader.
        """
        self.eval()
        in_scores = []
        out_scores = []

        for (x, _) in tqdm(in_loader, desc='In-distribution', leave=False):
            x = x.to(self.device)
            z, sldj = self.forward(x)
            in_scores.append(self.nll_scores(z, sldj).cpu().numpy())
            

        for (x, _) in tqdm(out_loader, desc='Out-of-distribution', leave=False):
            x = x.to(self.device)
            z, sldj = self.forward(x)
            out_scores.append(self.nll_scores(z, sldj).cpu().numpy())
            

        in_scores = np.concatenate(in_scores)
        out_scores = np.concatenate(out_scores)

        # Plot histogram of scores
        plt.hist(in_scores, bins=50, alpha=0.5, label='In-distribution')
        plt.hist(out_scores, bins=50, alpha=0.5, label='Out-of-distribution')
        plt.legend()
        plt.show()



class _FlowStep(nn.Module):
    """Recursive builder for a Flow++ model.

    Each `_FlowStep` corresponds to a single scale in Flow++.
    The constructor is recursively called to build a full model.

    Args:
        scales (tuple): Number of each type of coupling layer in each scale.
            Each scale is a 2-tuple of the form (num_channelwise, num_checkerboard).
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        num_components (int): Number of components in the mixture.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, scales, in_shape, mid_channels, num_blocks, num_components, use_attn, drop_prob):
        super(_FlowStep, self).__init__()
        in_channels, in_height, in_width = in_shape
        num_channelwise, num_checkerboard = scales[0]
        channels = []
        for i in range(num_channelwise):
            channels += [ActNorm(in_channels // 2),
                         InvConv(in_channels // 2),
                         Coupling(in_channels=in_channels // 2,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob),
                         Flip()]

        checkers = []
        for i in range(num_checkerboard):
            checkers += [ActNorm(in_channels),
                         InvConv(in_channels),
                         Coupling(in_channels=in_channels,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob),
                         Flip()]
        self.channels = nn.ModuleList(channels) if channels else None
        self.checkers = nn.ModuleList(checkers) if checkers else None

        if len(scales) <= 1:
            self.next = None
        else:
            next_shape = (2 * in_channels, in_height // 2, in_width // 2)
            self.next = _FlowStep(scales=scales[1:],
                                  in_shape=next_shape,
                                  mid_channels=mid_channels,
                                  num_blocks=num_blocks,
                                  num_components=num_components,
                                  use_attn=use_attn,
                                  drop_prob=drop_prob)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if self.next is not None:
                x = squeeze(x)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)

            if self.checkers:
                x = checkerboard(x)
                for flow in reversed(self.checkers):
                    x, sldj = flow(x, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.channels:
                x = channelwise(x)
                for flow in reversed(self.channels):
                    x, sldj = flow(x, sldj, reverse)
                x = channelwise(x, reverse=True)
        else:
            if self.channels:
                x = channelwise(x)
                for flow in self.channels:
                    x, sldj = flow(x, sldj, reverse)
                x = channelwise(x, reverse=True)

            if self.checkers:
                x = checkerboard(x)
                for flow in self.checkers:
                    x, sldj = flow(x, sldj, reverse)
                x = checkerboard(x, reverse=True)

            if self.next is not None:
                x = squeeze(x)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = unsqueeze(x)

        return x, sldj


class _Dequantization(nn.Module):
    """Dequantization Network for Flow++

    Args:
        in_shape (int): Shape of the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
        use_attn (bool): Use attention in the coupling layers.
        drop_prob (float): Dropout probability.
        num_flows (int): Number of InvConv+MLCoupling flows to use.
        aux_channels (int): Number of channels in auxiliary input to couplings.
        num_components (int): Number of components in the mixture.
    """
    def __init__(self, in_shape, mid_channels, num_blocks, use_attn, drop_prob,
                 num_flows=4, aux_channels=32, num_components=32):
        super(_Dequantization, self).__init__()
        in_channels, in_height, in_width = in_shape
        self.aux_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, aux_channels, kernel_size=3, padding=1),
            GatedConv(aux_channels, drop_prob),
            GatedConv(aux_channels, drop_prob),
            GatedConv(aux_channels, drop_prob))

        flows = []
        for _ in range(num_flows):
            flows += [ActNorm(in_channels),
                      InvConv(in_channels),
                      Coupling(in_channels, mid_channels, num_blocks,
                               num_components, drop_prob,
                               use_attn=use_attn,
                               aux_channels=aux_channels),
                      Flip()]
        self.flows = nn.ModuleList(flows)

    def forward(self, x, sldj):
        u = torch.randn_like(x)
        eps_nll = 0.5 * (u ** 2 + math.log(2 * math.pi))

        aux = self.aux_conv(torch.cat(checkerboard(x - 0.5), dim=1))
        u = checkerboard(u)
        for i, flow in enumerate(self.flows):
            u, sldj = flow(u, sldj, aux=aux) if i % 4 == 2 else flow(u, sldj)
        u = checkerboard(u, reverse=True)

        u = torch.sigmoid(u)
        x = (x * 255. + u) / 256.

        sigmoid_ldj = safe_log(u) + safe_log(1. - u)
        sldj = sldj + (eps_nll + sigmoid_ldj).flatten(1).sum(-1)

        return x, sldj
    

class _BaseNorm(nn.Module):
    """Base class for ActNorm (Glow) and PixNorm (Flow++).

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_channels, height, width):
        super(_BaseNorm, self).__init__()

        # Input gets concatenated along channel axis
        num_channels *= 2

        self.register_buffer('is_initialized', torch.zeros(1))
        self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.inv_std = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.eps = 1e-6

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            mean, inv_std = self._get_moments(x)
            self.mean.data.copy_(mean.data)
            self.inv_std.data.copy_(inv_std.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _get_moments(self, x):
        raise NotImplementedError('Subclass of _BaseNorm must implement _get_moments')

    def _scale(self, x, sldj, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, ldj=None, reverse=False):
        x = torch.cat(x, dim=1)
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        x = x.chunk(2, dim=1)

        return x, ldj


class ActNorm(_BaseNorm):
    """Activation Normalization used in Glow

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_channels):
        super(ActNorm, self).__init__(num_channels, 1, 1)

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum() * x.size(2) * x.size(3)
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum() * x.size(2) * x.size(3)

        return x, sldj


class PixNorm(_BaseNorm):
    """Pixel-wise Activation Normalization used in Flow++

    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    initialization, `mean` and `inv_std` become trainable parameters.
    """
    def _get_moments(self, x):
        mean = torch.mean(x.clone(), dim=0, keepdim=True)
        var = torch.mean((x.clone() - mean) ** 2, dim=0, keepdim=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum()
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum()

        return x, sldj
    
def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.

    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.

    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor
    

class Coupling(nn.Module):
    """Mixture-of-Logistics Coupling layer in Flow++

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the transformation network.
        num_blocks (int): Number of residual blocks in the transformation network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in the NN blocks.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, num_components, drop_prob,
                 use_attn=True, aux_channels=None):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, mid_channels, num_blocks, num_components, drop_prob, use_attn, aux_channels)

    def forward(self, x, sldj=None, reverse=False, aux=None):
        x_change, x_id = x
        a, b, pi, mu, s = self.nn(x_id, aux)

        if reverse:
            out = x_change * a.mul(-1).exp() - b
            out, scale_ldj = inverse(out, reverse=True)
            out = out.clamp(1e-5, 1. - 1e-5)
            out = mixture_inv_cdf(out, pi, mu, s)
            logistic_ldj = mixture_log_pdf(out, pi, mu, s)
            sldj = sldj - (a + scale_ldj + logistic_ldj).flatten(1).sum(-1)
        else:
            out = mixture_log_cdf(x_change, pi, mu, s).exp()
            out, scale_ldj = inverse(out)
            out = (out + b) * a.exp()
            logistic_ldj = mixture_log_pdf(x_change, pi, mu, s)
            sldj = sldj + (logistic_ldj + scale_ldj + a).flatten(1).sum(-1)

        x = (out, x_id)

        return x, sldj
    
def _log_pdf(x, mean, log_scale):
    """Element-wise log density of the logistic distribution."""
    z = (x - mean) * torch.exp(-log_scale)
    log_p = z - log_scale - 2 * F.softplus(z)

    return log_p


def _log_cdf(x, mean, log_scale):
    """Element-wise log CDF of the logistic distribution."""
    z = (x - mean) * torch.exp(-log_scale)
    log_p = F.logsigmoid(z)

    return log_p


def mixture_log_pdf(x, prior_logits, means, log_scales):
    """Log PDF of a mixture of logistic distributions."""
    log_ps = F.log_softmax(prior_logits, dim=1) \
        + _log_pdf(x.unsqueeze(1), means, log_scales)
    log_p = torch.logsumexp(log_ps, dim=1)

    return log_p


def mixture_log_cdf(x, prior_logits, means, log_scales):
    """Log CDF of a mixture of logistic distributions."""
    log_ps = F.log_softmax(prior_logits, dim=1) \
        + _log_cdf(x.unsqueeze(1), means, log_scales)
    log_p = torch.logsumexp(log_ps, dim=1)

    return log_p


def mixture_inv_cdf(y, prior_logits, means, log_scales,
                    eps=1e-10, max_iters=100):
    """Inverse CDF of a mixture of logisitics. Iterative algorithm."""
    if y.min() <= 0 or y.max() >= 1:
        raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')

    def body(x_, lb_, ub_):
        cur_y = torch.exp(mixture_log_cdf(x_, prior_logits, means,
                                          log_scales))
        gt = (cur_y > y).type(y.dtype)
        lt = 1 - gt
        new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
        new_lb = gt * lb_ + lt * x_
        new_ub = gt * x_ + lt * ub_
        return new_x_, new_lb, new_ub

    x = torch.zeros_like(y)
    max_scales = torch.sum(torch.exp(log_scales), dim=1, keepdim=True)
    lb, _ = (means - 20 * max_scales).min(dim=1)
    ub, _ = (means + 20 * max_scales).max(dim=1)
    diff = float('inf')

    i = 0
    while diff > eps and i < max_iters:
        new_x, lb, ub = body(x, lb, ub)
        diff = (new_x - x).abs().max()
        x = new_x
        i += 1

    return x


def inverse(x, reverse=False):
    """Inverse logistic function."""
    if reverse:
        z = torch.sigmoid(x)
        ldj = - F.softplus(x) - F.softplus(-x) ####### dont know if its wrong
    else:
        z = -safe_log(x.reciprocal() - 1.)
        ldj = -safe_log(x) - safe_log(1. - x)

    return z, ldj

class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
        random_init (bool): Initialize with a random orthogonal matrix.
            Otherwise initialize with noisy identity.
    """
    def __init__(self, num_channels, random_init=False):
        super(InvConv, self).__init__()
        self.num_channels = 2 * num_channels

        if random_init:
            # Initialize with a random orthogonal matrix
            w_init = np.random.randn(self.num_channels, self.num_channels)
            w_init = np.linalg.qr(w_init)[0]
        else:
            # Initialize as identity permutation with some noise
            w_init = np.eye(self.num_channels, self.num_channels) \
                     + 1e-3 * np.random.randn(self.num_channels, self.num_channels)
        self.weight = nn.Parameter(torch.from_numpy(w_init.astype(np.float32)))

    def forward(self, x, sldj, reverse=False):
        x = torch.cat(x, dim=1)

        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        x = F.conv2d(x, weight)
        x = x.chunk(2, dim=1)

        return x, sldj
    
class NN(nn.Module):
    """Neural network used to parametrize the transformations of an MLCoupling.

    An `NN` is a stack of blocks, where each block consists of the following
    two layers connected in a residual fashion:
      1. Conv: input -> nonlinearit -> conv3x3 -> nonlinearity -> gate
      2. Attn: input -> conv1x1 -> multihead self-attention -> gate,
    where gate refers to a 1×1 convolution that doubles the number of channels,
    followed by a gated linear unit (Dauphin et al., 2016).
    The convolutional layer is identical to the one used by PixelCNN++
    (Salimans et al., 2017), and the multi-head self attention mechanism we
    use is identical to the one in the Transformer (Vaswani et al., 2017).

    Args:
        in_channels (int): Number of channels in the input.
        num_channels (int): Number of channels in each block of the network.
        num_blocks (int): Number of blocks in the network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in each block.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, in_channels, num_channels, num_blocks, num_components, drop_prob, use_attn=True, aux_channels=None):
        super(NN, self).__init__()
        self.k = num_components  # k = number of mixture components
        self.in_conv = WNConv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.mid_convs = nn.ModuleList([ConvAttnBlock(num_channels, drop_prob, use_attn, aux_channels)
                                        for _ in range(num_blocks)])
        self.out_conv = WNConv2d(num_channels, in_channels * (2 + 3 * self.k),
                                 kernel_size=3, padding=1)
        self.rescale = weight_norm(Rescale(in_channels))

    def forward(self, x, aux=None):
        b, c, h, w = x.size()
        x = self.in_conv(x)
        for conv in self.mid_convs:
            x = conv(x, aux)
        x = self.out_conv(x)

        # Split into components and post-process
        x = x.view(b, -1, c, h, w)
        s, t, pi, mu, scales = x.split((1, 1, self.k, self.k, self.k), dim=1)
        s = self.rescale(torch.tanh(s.squeeze(1)))
        t = t.squeeze(1)
        scales = scales.clamp(min=-7)  # From the code in original Flow++ paper

        return s, t, pi, mu, scales


class ConvAttnBlock(nn.Module):
    def __init__(self, num_channels, drop_prob, use_attn, aux_channels):
        super(ConvAttnBlock, self).__init__()
        self.conv = GatedConv(num_channels, drop_prob, aux_channels)
        self.norm_1 = nn.LayerNorm(num_channels)
        if use_attn:
            self.attn = GatedAttn(num_channels, drop_prob=drop_prob)
            self.norm_2 = nn.LayerNorm(num_channels)
        else:
            self.attn = None

    def forward(self, x, aux=None):
        x = self.conv(x, aux) + x
        x = x.permute(0, 2, 3, 1)  # (b, h, w, c)
        x = self.norm_1(x)

        if self.attn:
            x = self.attn(x) + x
            x = self.norm_2(x)
        x = x.permute(0, 3, 1, 2)  # (b, c, h, w)

        return x


class GatedAttn(nn.Module):
    """Gated Multi-Head Self-Attention Block

    Based on the paper:
    "Attention Is All You Need"
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
        Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    (https://arxiv.org/abs/1706.03762).

    Args:
        d_model (int): Number of channels in the input.
        num_heads (int): Number of attention heads.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, d_model, num_heads=4, drop_prob=0.):
        super(GatedAttn, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.in_proj = weight_norm(nn.Linear(d_model, 3 * d_model, bias=False))
        self.gate = weight_norm(nn.Linear(d_model, 2 * d_model))

    def forward(self, x):
        # Flatten and encode position
        b, h, w, c = x.size()
        x = x.view(b, h * w, c)
        _, seq_len, num_channels = x.size()
        pos_encoding = self.get_pos_enc(seq_len, num_channels, x.device)
        x = x + pos_encoding

        # Compute q, k, v
        memory, query = torch.split(self.in_proj(x), (2 * c, c), dim=-1)
        q = self.split_last_dim(query, self.num_heads)
        k, v = [self.split_last_dim(tensor, self.num_heads)
                for tensor in torch.split(memory, self.d_model, dim=2)]
        # Compute attention and reshape
        key_depth_per_head = self.d_model // self.num_heads
        q_c = q.clone() * (key_depth_per_head ** -0.5)
        x = self.dot_product_attention(q_c, k, v)
        x = self.combine_last_two_dim(x.permute(0, 2, 1, 3))
        x = x.transpose(1, 2).view(b, c, h, w).permute(0, 2, 3, 1)  # (b, h, w, c)

        x = self.gate(x)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)

        return x

    def dot_product_attention(self, q, k, v, bias=False):
        """Dot-product attention.

        Args:
            q (torch.Tensor): Queries of shape (batch, heads, length_q, depth_k)
            k (torch.Tensor): Keys of shape (batch, heads, length_kv, depth_k)
            v (torch.Tensor): Values of shape (batch, heads, length_kv, depth_v)
            bias (bool): Use bias for attention.

        Returns:
            attn (torch.Tensor): Output of attention mechanism.
        """
        weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            weights += self.bias
        weights = F.softmax(weights, dim=-1)
        weights = F.dropout(weights, self.drop_prob, self.training)
        attn = torch.matmul(weights, v)

        return attn

    @staticmethod
    def split_last_dim(x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x (torch.Tensor): Tensor with shape (..., m)
            n (int): Size of second-to-last dimension.
        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., n, m/n)
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)

        return ret.permute(0, 2, 1, 3)

    @staticmethod
    def combine_last_two_dim(x):
        """Merge the last two dimensions of `x`.

        Args:
            x (torch.Tensor): Tensor with shape (..., m, n)

        Returns:
            ret (torch.Tensor): Resulting tensor with shape (..., m * n)
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)

        return ret

    @staticmethod
    def get_pos_enc(seq_len, num_channels, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        num_timescales = num_channels // 2
        log_timescale_increment = math.log(10000.) / (num_timescales - 1)
        inv_timescales = torch.arange(num_timescales,
                                      dtype=torch.float32,
                                      device=device)
        inv_timescales *= -log_timescale_increment
        inv_timescales = inv_timescales.exp_()
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        encoding = torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
        encoding = F.pad(encoding, [0, num_channels % 2, 0, 0])
        encoding = encoding.view(1, seq_len, num_channels)

        return encoding


class GatedConv(nn.Module):
    """Gated Convolution Block

    Originally used by PixelCNN++ (https://arxiv.org/pdf/1701.05517).

    Args:
        num_channels (int): Number of channels in hidden activations.
        drop_prob (float): Dropout probability.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, num_channels, drop_prob=0., aux_channels=None):
        super(GatedConv, self).__init__()
        self.nlin = concat_elu
        self.conv = WNConv2d(2 * num_channels, num_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(drop_prob)
        self.gate = WNConv2d(2 * num_channels, 2 * num_channels, kernel_size=1, padding=0)
        if aux_channels is not None:
            self.aux_conv = WNConv2d(2 * aux_channels, num_channels, kernel_size=1, padding=0)
        else:
            self.aux_conv = None

    def forward(self, x, aux=None):
        x = self.nlin(x)
        x = self.conv(x)
        if aux is not None:
            aux = self.nlin(aux)
            x = x + self.aux_conv(aux)
        x = self.nlin(x)
        x = self.drop(x)
        x = self.gate(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)

        return x


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
    
############################## Utils ##############################

class Flip(nn.Module):
    def forward(self, x, sldj, reverse=False):
        assert isinstance(x, tuple) and len(x) == 2
        return (x[1], x[0]), sldj


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.

    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.

    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def checkerboard(x, reverse=False):
    """Split x in a checkerboard pattern. Collapse horizontally."""
    # Get dimensions
    if reverse:
        b, c, h, w = x[0].size()
        w *= 2
        device = x[0].device
    else:
        b, c, h, w = x.size()
        device = x.device

    # Get list of indices in alternating checkerboard pattern
    y_idx = []
    z_idx = []
    for i in range(h):
        for j in range(w):
            if (i % 2) == (j % 2):
                y_idx.append(i * w + j)
            else:
                z_idx.append(i * w + j)
    y_idx = torch.tensor(y_idx, dtype=torch.int64, device=device)
    z_idx = torch.tensor(z_idx, dtype=torch.int64, device=device)

    if reverse:
        y, z = (t.contiguous().view(b, c, h * w // 2) for t in x)
        x = torch.zeros(b, c, h * w, dtype=y.dtype, device=y.device)
        x[:, :, y_idx] += y
        x[:, :, z_idx] += z
        x = x.view(b, c, h, w)

        return x
    else:
        if w % 2 != 0:
            raise RuntimeError('Checkerboard got odd width input: {}'.format(w))

        x = x.view(b, c, h * w)
        y = x[:, :, y_idx].view(b, c, h, w // 2)
        z = x[:, :, z_idx].view(b, c, h, w // 2)

        return y, z


def channelwise(x, reverse=False):
    """Split x channel-wise."""
    if reverse:
        x = torch.cat(x, dim=1)
        return x
    else:
        y, z = x.chunk(2, dim=1)
        return y, z


def squeeze(x):
    """Trade spatial extent for channels. I.e., convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x


def unsqueeze(x):
    """Trade channels channels for spatial extent. I.e., convert each
    4x1x1 volume of input into a 1x4x4 volume of output.

    Args:
        x (torch.Tensor): Input to unsqueeze.

    Returns:
        x (torch.Tensor): Unsqueezed tensor.
    """
    b, c, h, w = x.size()
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 4, h * 2, w * 2)

    return x


def concat_elu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU."""
    return F.elu(torch.cat((x, -x), dim=1))


def safe_log(x):
    return torch.log(x.clamp(min=1e-22))

def get_param_groups(net, weight_decay, norm_suffix='weight_g', verbose=False):
    """Get two parameter groups from `net`: One named "normalized" which will
    override the optimizer with `weight_decay`, and one named "unnormalized"
    which will inherit all hyperparameters from the optimizer.
    Args:
        net (torch.nn.Module): Network to get parameters from
        weight_decay (float): Weight decay to apply to normalized weights.
        norm_suffix (str): Suffix to select weights that should be normalized.
            For WeightNorm, using 'weight_g' normalizes the scale variables.
        verbose (bool): Print out number of normalized and unnormalized parameters.
    """
    norm_params = []
    unnorm_params = []
    for n, p in net.named_parameters():
        if n.endswith(norm_suffix):
            norm_params.append(p)
        else:
            unnorm_params.append(p)

    param_groups = [{'name': 'normalized', 'params': norm_params, 'weight_decay': weight_decay},
                    {'name': 'unnormalized', 'params': unnorm_params}]

    if verbose:
        print('{} normalized parameters'.format(len(norm_params)))
        print('{} unnormalized parameters'.format(len(unnorm_params)))

    return param_groups


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)

        return x
    
def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)


class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll
    
class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count