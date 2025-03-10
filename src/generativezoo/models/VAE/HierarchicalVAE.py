#########################################################
### Code based on: https://github.com/GlassyWing/nvae ###
#########################################################

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from functools import reduce
import robust_loss_pytorch
from tqdm import tqdm, trange
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import os
from config import models_dir
import wandb

class WarmupKLLoss:

    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        """
        Warm-up loss for KL divergence
        :param init_weights: List. Initial weights for each stage
        :param steps: List. Steps for each stage
        :param M_N: float. Maximum value for KL divergence
        :param eta_M_N: float. Minimum value for KL divergence
        :param M_N_decay_step: int. Decay steps for KL divergence
        """
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:

            if self.stage > len(self.steps) - 1:
                break

            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        '''
        Get KL loss with warm-up
        :param step: int. Current step
        :param losses: List. KL losses for each stage
        :return: Tensor. KL loss
        '''
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            M_N = self.M_N

        return M_N * loss


def add_sn(m):
    '''
    Add spectral normalization to module
    :param m: nn.Module
    :return: nn.Module
    '''
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def random_uniform_like(tensor, min_val, max_val):
    return (max_val - min_val) * torch.rand_like(tensor) + min_val


def sample_from_discretized_mix_logistic(y, img_channels=3, log_scale_min=-7.):
    """

    :param y: Tensor, shape=(batch_size, 3 * num_mixtures * img_channels, height, width),
    :return: Tensor: sample in range of [-1, 1]
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y.chunk(3, dim=1)

    temp = random_uniform_like(logit_probs, min_val=1e-5, max_val=1. - 1e-5)
    temp = logit_probs - torch.log(-torch.log(temp))

    ones = torch.eye(means.size(1) // img_channels, dtype=means.dtype, device=means.device)

    sample = []
    for logit_prob, mean, log_scale, tmp in zip(logit_probs.chunk(img_channels, dim=1),
                                                means.chunk(img_channels, dim=1),
                                                log_scales.chunk(img_channels, dim=1),
                                                temp.chunk(img_channels, dim=1)):
        # (batch_size, height, width)
        argmax = torch.max(tmp, dim=1)[1]
        B, H, W = argmax.shape

        one_hot = ones.index_select(0, argmax.flatten())
        one_hot = one_hot.view(B, H, W, mean.size(1)).permute(0, 3, 1, 2).contiguous()

        # (batch_size, 1, height, width)
        mean = torch.sum(mean * one_hot, dim=1)
        log_scale = torch.clamp_max(torch.sum(log_scale * one_hot, dim=1), log_scale_min)

        u = random_uniform_like(mean, min_val=1e-5, max_val=1. - 1e-5)
        x = mean + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))
        sample.append(x)

    # (batch_size, img_channels, height, width)
    sample = torch.stack(sample, dim=1)

    return sample

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._seq = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)


# class FourierMapping(nn.Module):
#
#     def __init__(self, dims, seed):
#         super().__init__()
#         np.random.seed(seed)
#         B = np.random.randn(*dims) * 10
#         np.random.seed(None)
#         self.B = torch.tensor(B, dtype=torch.float32)
#
#     def forward(self, x):
#         x = input_mapping(x, self.B.to(x.device))
#         return x


class EncoderResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.seq = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self.seq(x)


class DecoderResidualBlock(nn.Module):

    def __init__(self, dim, n_group):
        super().__init__()

        self._seq = nn.Sequential(
            nn.Conv2d(dim, n_group * dim, kernel_size=1),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, n_group * dim, kernel_size=5, padding=2, groups=n_group),
            nn.BatchNorm2d(n_group * dim), Swish(),
            nn.Conv2d(n_group * dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            SELayer(dim))

    def forward(self, x):
        return x + 0.1 * self._seq(x)
    
class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1),
            nn.BatchNorm2d(out_channel // 2), Swish(),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel), Swish()
        )

    def forward(self, x):
        return self._seq(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(ConvBlock(channels[i], channels[i + 1]))

        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class Encoder(nn.Module):

    def __init__(self, z_dim, channels=3):
        '''
        Encoder for Hierarchical VAE
        :param z_dim: int. Dimension of latent space
        :param channels: int. Number of channels of input image
        '''
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([channels, z_dim // 16, z_dim // 8]),  # (16, 16)
            EncoderBlock([z_dim // 8, z_dim // 4, z_dim // 2]),  # (4, 4)
            EncoderBlock([z_dim // 2, z_dim]),  # (2, 2)
        ])

        self.encoder_residual_blocks = nn.ModuleList([
            EncoderResidualBlock(z_dim // 8),
            EncoderResidualBlock(z_dim // 2),
            EncoderResidualBlock(z_dim),
        ])

        self.condition_x = nn.Sequential(
            Swish(),
            nn.Conv2d(z_dim, z_dim * 2, kernel_size=1)
        )

    def forward(self, x):
        xs = []
        last_x = x
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x
            xs.append(x)

        mu, log_var = self.condition_x(last_x).chunk(2, dim=1)

        return mu, log_var, xs[:-1][::-1]
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(

            nn.ConvTranspose2d(in_channel,
                               out_channel,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel), Swish(),
        )

    def forward(self, x):
        return self._seq(x)


def recon(output, target):
    """
    recon loss
    :param output: Tensor. shape = (B, C, H, W)
    :param target: Tensor. shape = (B, C, H, W)
    :return:
    """

    # Treat q(x|z) as Norm distribution
    # loss = F.mse_loss(output, target)

    # Treat q(x|z) as Bernoulli distribution.
    loss = F.binary_cross_entropy(output, target)
    return loss


def kl(mu, log_var):
    """
    kl loss with standard norm distribute
    :param mu:
    :param log_var:
    :return:
    """
    loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def kl_2(delta_mu, delta_log_var, mu, log_var):
    var = torch.exp(log_var)
    delta_var = torch.exp(delta_log_var)

    loss = -0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / var - delta_var, dim=[1, 2, 3])
    return torch.mean(loss, dim=0)


def log_sum_exp(x):
    """

    :param x: Tensor. shape = (batch_size, num_mixtures, height, width)
    :return:
    """

    m2 = torch.max(x, dim=1, keepdim=True)[0]
    m = m2.unsqueeze(1)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=1))


def discretized_mix_logistic_loss(y_hat: torch.Tensor, y: torch.Tensor, num_classes=256, log_scale_min=-7.0):
    """Discretized mix of logistic distributions loss.

    Note that it is assumed that input is scaled to [-1, 1]



    :param y_hat: Tensor. shape=(batch_size, 3 * num_mixtures * img_channels, height, width), predict output.
    :param y: Tensor. shape=(batch_size, img_channels, height, width), Target.
    :return: Tensor loss
    """

    # unpack parameters, [batch_size, num_mixtures * img_channels, height, width] x 3
    logit_probs, means, log_scales = y_hat.chunk(3, dim=1)
    log_scales = torch.clamp_max(log_scales, log_scale_min)

    num_mixtures = y_hat.size(1) // y.size(1) // 3

    B, C, H, W = y.shape
    y = y.unsqueeze(1).repeat(1, num_mixtures, 1, 1, 1).permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    log_pdf_mid = min_in - log_scales - 2. * F.softplus(mid_in)

    log_probs = torch.where(y < -0.999, log_cdf_plus,
                            torch.where(y > 0.999, log_one_minus_cdf_min,
                                        torch.where(cdf_delta > 1e-5, torch.clamp_max(cdf_delta, 1e-12),
                                                    log_pdf_mid - np.log((num_classes - 1) / 2))))

    # (batch_size, num_mixtures * img_channels, height, width)
    log_probs = log_probs + F.softmax(log_probs, dim=1)

    log_probs = [log_sum_exp(log_prob) for log_prob in log_probs.chunk(y.size(1), dim=1)]
    log_probs = reduce(lambda a, b: a + b, log_probs)

    return -torch.sum(log_probs)

class DecoderBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(UpsampleBlock(channels[i], channels[i + 1]))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class Decoder(nn.Module):

    def __init__(self, z_dim, channels=3):
        '''
        Decoder for Hierarchical VAE
        :param z_dim: int. Dimension of latent space
        :param channels: int. Number of channels of input image
        '''
        super().__init__()

        # Input channels = z_channels * 2 = x_channels + z_channels
        # Output channels = z_channels
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock([z_dim * 2, z_dim // 2]),  # 2x upsample
            DecoderBlock([z_dim, z_dim // 4, z_dim // 8]),  # 4x upsample
            DecoderBlock([z_dim // 4, z_dim // 16, z_dim // 32])  # 4x uplsampe
        ])
        self.decoder_residual_blocks = nn.ModuleList([
            DecoderResidualBlock(z_dim // 2, n_group=4),
            DecoderResidualBlock(z_dim // 8, n_group=2),
            DecoderResidualBlock(z_dim // 32, n_group=1)
        ])

        # p(z_l | z_(l-1))
        self.condition_z = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim // 2),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 8),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        # p(z_l | x, z_(l-1))
        self.condition_xz = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(z_dim),
                nn.Conv2d(z_dim, z_dim // 2, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 2, z_dim, kernel_size=1)
            ),
            nn.Sequential(
                ResidualBlock(z_dim // 4),
                nn.Conv2d(z_dim // 4, z_dim // 8, kernel_size=1),
                Swish(),
                nn.Conv2d(z_dim // 8, z_dim // 4, kernel_size=1)
            )
        ])

        self.recon = nn.Sequential(
            ResidualBlock(z_dim // 32),
            nn.Conv2d(z_dim // 32, channels, kernel_size=1),
        )

        self.zs = []

    def forward(self, z, xs=None, mode="random", freeze_level=-1):
        """

        :param z: shape. = (B, z_dim, map_h, map_w)
        :return:
        """

        B, D, map_h, map_w = z.shape

        # The init h (hidden state), can be replace with learned param, but it didn't work much
        decoder_out = torch.zeros(B, D, map_h, map_w, device=z.device, dtype=z.dtype)

        kl_losses = []
        if freeze_level != -1 and len(self.zs) == 0 :
            self.zs.append(z)

        for i in range(len(self.decoder_residual_blocks)):

            z_sample = torch.cat([decoder_out, z], dim=1)
            decoder_out = self.decoder_residual_blocks[i](self.decoder_blocks[i](z_sample))

            if i == len(self.decoder_residual_blocks) - 1:
                break

            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

            if xs is not None:
                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs[i], decoder_out], dim=1)) \
                    .chunk(2, dim=1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))
                mu = mu + delta_mu
                log_var = log_var + delta_log_var

            if mode == "fix" and i < freeze_level:
                if len(self.zs) < freeze_level + 1:
                    z = reparameterize(mu, 0)
                    self.zs.append(z)
                else:
                    z = self.zs[i + 1]
            elif mode == "fix":
                z = reparameterize(mu, 0 if i == 0 else torch.exp(0.5 * log_var))
            else:
                z = reparameterize(mu, torch.exp(0.5 * log_var))

            map_h *= 2 ** (len(self.decoder_blocks[i].channels) - 1)
            map_w *= 2 ** (len(self.decoder_blocks[i].channels) - 1)

        x_hat = torch.sigmoid(self.recon(decoder_out))

        return x_hat, kl_losses
    
def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, "HierarchicalVAE")):
        os.makedirs(os.path.join(models_dir, "HierarchicalVAE"))
    
class HierarchicalVAE(nn.Module):

    def __init__(self, z_dim, img_dim, channels=3, no_wandb=False):
        '''
        Hierarchical VAE
        :param z_dim: int. Dimension of latent space
        :param img_dim: tuple. (H, W) of input image
        :param channels: int. Number of channels of input image
        '''
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(z_dim, channels).to(self.device)
        self.decoder = Decoder(z_dim, channels).to(self.device)

        self.img_dim = img_dim[0]
        self.channels = channels
        self.z_dim = z_dim
        self.no_wandb = no_wandb

        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=1, float_dtype=np.float32, device="cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """

        :param x: Tensor. shape = (B, C, H, W)
        :return:
        """

        mu, log_var, xs = self.encoder(x)

        # (B, D_Z)
        z = reparameterize(mu, torch.exp(0.5 * log_var))

        decoder_output, losses = self.decoder(z, xs)

        # Treat p(x|z) as discretized_mix_logistic distribution cost so much, this is an alternative way
        # witch combine multi distribution.
        recon_loss = torch.mean(self.adaptive_loss.lossfun(
            torch.mean(F.binary_cross_entropy(decoder_output, x, reduction='none'), dim=[1, 2, 3])[:, None]))

        kl_loss = kl(mu, log_var)

        return decoder_output, recon_loss, [kl_loss] + losses

    def train_model(self, data_loader, args, verbose=True):
        """

        :param data_loader:
        :param args:
        :return:
        """

        warmup_kl = WarmupKLLoss(init_weights=[1., 1. / 2, 1. / 8],
                             steps=[4500, 3000, 1500],
                             M_N=args.batch_size / len(data_loader.dataset),
                             eta_M_N=5e-6,
                             M_N_decay_step=36000)
        
        print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)
        create_checkpoint_dir()

        optimizer = torch.optim.Adamax(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)
        self.apply(add_sn)
        epoch_bar = trange(args.n_epochs, desc="Epoch")
        step = 0

        best_loss = np.inf

        for epoch in epoch_bar:

            self.train()
            epoch_loss = 0.
            epoch_recon_loss = 0.
            epoch_kl_loss = 0.

            for x,_ in tqdm(data_loader, desc="Train", leave=False, disable=not verbose):
                x = x.to(self.device)
                optimizer.zero_grad()

                decoder_output, recon_loss, kl_losses = self.forward(x)

                kl_loss = warmup_kl.get_loss(step, kl_losses)
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()*x.size(0)
                epoch_recon_loss += recon_loss.item()*x.size(0)
                epoch_kl_loss += kl_loss.item()*x.size(0)

                step += 1

            epoch_loss /= len(data_loader.dataset)
            epoch_recon_loss /= len(data_loader.dataset)
            epoch_kl_loss /= len(data_loader.dataset)

            if not self.no_wandb:
                wandb.log({"loss": epoch_loss, "recon_loss": epoch_recon_loss, "kl_loss": epoch_kl_loss})

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # save encoder and decoder state dicts
                dict_to_save = {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}
                torch.save(dict_to_save, os.path.join(models_dir, "HierarchicalVAE", f"HVAE_{args.dataset}.pt"))

            epoch_bar.set_postfix(loss=epoch_loss, recon_loss=epoch_recon_loss, kl_loss=epoch_kl_loss)
            scheduler.step()

            if epoch == 0 or (epoch + 1) % 5 == 0:
                self.sample(16, train=True)

    def load_checkpoint(self, checkpoint):
        """

        :param checkpoint:
        :return:
        """

        # load encoder and decoder state dicts
        checkpoint = torch.load(checkpoint, weights_only=False)
        self.apply(add_sn)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])



    @torch.no_grad()
    def sample(self, num_samples, train=False):
        """

        :param num_samples:
        :return:
        """

        self.eval()
        z = torch.randn(num_samples, self.z_dim, self.img_dim//32, self.img_dim//32, device=self.device)
        decoder_output, _ = self.decoder(z)

        fig = plt.figure(figsize=(10, 10))
        grid = make_grid(decoder_output, nrow=int(num_samples ** 0.5))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        if train:
            if not self.no_wandb:
                wandb.log({"train_samples": fig})
        else:
            plt.show()
        