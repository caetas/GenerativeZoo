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
from sklearn.metrics import roc_auc_score
from config import models_dir
from torchvision.transforms import Compose, Lambda, ToPILImage

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'VanillaDDPM')):
    os.makedirs(os.path.join(models_dir, 'VanillaDDPM'))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Attention(nn.Module):
    def __init__(self, num_channels, num_heads=4, head_dim=32):
        '''
        Attention module
        :param num_channels: number of channels in the input image
        :param num_heads: number of heads in the multi-head attention
        :param head_dim: dimension of each head
        '''
        super().__init__()
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads
        self.to_qkv = nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=1)
        
    def forward(self, x):
        '''
        Forward pass of the attention module
        :param x: input image
        '''
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        '''
        Block module
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param groups: number of groups for group normalization
        '''
        super().__init__()
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(num_gruops=groups, num_channels=out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        '''
        Forward pass of the block module
        :param x: input image
        :param scale_shift: scale and shift values
        '''
        x = self.projection(x)
        x = self.group_norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_embedding_dim=None, channel_scale_factor=2, normalize=True):
        '''
        ConvNextBlock module
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param time_embedding_dim: dimension of the time embedding
        :param channel_scale_factor: scaling factor for the number of channels
        :param normalize: whether to normalize the output
        '''
        super().__init__()
        self.time_projection = (
            nn.Sequential(
                nn.GELU(), 
                nn.Linear(in_features=time_embedding_dim, out_features=in_channels)
            )
            if exists(x=time_embedding_dim)
            else None
        )

        self.ds_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3, groups=in_channels))

        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels) if normalize else nn.Identity(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * channel_scale_factor, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=1, num_channels=out_channels * channel_scale_factor), 
            nn.Conv2d(in_channels=out_channels * channel_scale_factor, out_channels=out_channels, kernel_size=3, padding=1),
        )

        self.residual_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        '''
        Forward pass of the ConvNextBlock module
        :param x: input image
        :param time_emb: time embedding
        '''
        h = self.ds_conv(x)
        if exists(x=self.time_projection) and exists(x=time_emb):
            assert exists(x=time_emb), "time embedding must be passed in"
            condition = self.time_projection(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        

        h = self.net(h)
        return h + self.residual_connection(x)
    
class Downsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class LinearAttention(nn.Module):
    def __init__(self, num_channels, num_heads=4, head_dim=32):
        '''
        LinearAttention module
        :param num_channels: number of channels in the input image
        :param num_heads: number of heads in the multi-head attention
        :param head_dim: dimension of each head
        '''
        super().__init__()
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads
        self.to_qkv = nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim * 3, kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=1), 
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
        )

    def forward(self, x):
        '''
        Forward pass of the linear attention module
        :param x: input image
        '''
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = einsum("b h d n, b h e n -> b h d e", k, v)

        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.num_heads, x=h, y=w)
        return self.to_out(out)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        '''
        SinusoidalPositionEmbeddings module
        :param dim: dimension of the sinusoidal position embeddings
        '''
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.partial_embeddings = math.log(10000) / (self.half_dim - 1)
        
    
    def forward(self, time):
        device = time.device 
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * -self.partial_embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PreNorm(nn.Module):
    def __init__(self, num_channels, fn):
        super().__init__()
        self.fn = fn
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=num_channels)

    def forward(self, x):
        x = self.group_norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class ResNetBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, *, time_embedding_dim=None, groups=8):
        '''
        ResNetBlock module
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param time_embedding_dim: dimension of the time embedding
        :param groups: number of groups for group normalization
        '''
        super().__init__()
        self.time_projection = (
            nn.Sequential(
                nn.SiLU(), 
                nn.Linear(in_features=time_embedding_dim, out_features=out_channels) 
            )
            if exists(x=time_embedding_dim)
            else None
        )

        self.block1 = Block(in_channels=in_channels, out_channels=out_channels, groups=groups)
        self.block2 = Block(in_channels=out_channels, out_channels=out_channels, groups=groups)
        self.residual_connection = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(x=self.time_projection) and exists(x=time_emb):
            assert exists(x=time_emb), "time embedding must be passed in"
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.residual_connection(x)
    
class Upsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_features, init_channels=None, out_channels=None, channel_scale_factors=(1, 2, 4, 8), in_channels=3, with_time_emb=True, resnet_block_groups=8, use_convnext=True, convnext_scale_factor=2):
        '''
        UNet module
        :param n_features: number of features
        :param init_channels: number of initial channels
        :param out_channels: number of output channels
        :param channel_scale_factors: scaling factors for the number of channels
        :param in_channels: number of input channels
        :param with_time_emb: whether to use time embeddings
        :param resnet_block_groups: number of groups for group normalization in the ResNet block
        :param use_convnext: whether to use ConvNext block
        :param convnext_scale_factor: scaling factor for the number of channels in the ConvNext block
        '''
        super().__init__()

        # determine dimensions
        self.in_channels = in_channels

        init_channels = default(init_channels, n_features // 3 * 2)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=7, padding=3)

        dims = [init_channels, *map(lambda m: n_features * m, channel_scale_factors)]
        resolution_translations = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, channel_scale_factor=convnext_scale_factor)
        else:
            block_klass = partial(ResNetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = n_features * 4
            self.time_projection = nn.Sequential(
                SinusoidalPositionEmbeddings(dim=n_features),
                nn.Linear(in_features=n_features, out_features=time_dim),
                nn.GELU(),
                nn.Linear(in_features=time_dim, out_features=time_dim),
            )
        else:
            time_dim = None
            self.time_projection = None

        # layers
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        num_resolutions = len(resolution_translations)

        for idx, (in_chan, out_chan) in enumerate(resolution_translations):
            is_last = idx >= (num_resolutions - 1)
            self.encoder.append(
                nn.ModuleList(
                    [
                        block_klass(in_channels=in_chan, out_channels=out_chan, time_embedding_dim=time_dim),
                        block_klass(in_channels=out_chan, out_channels=out_chan, time_embedding_dim=time_dim),
                        Residual(fn=PreNorm(num_channels=out_chan, fn=LinearAttention(num_channels=out_chan))),
                        Downsample(num_channels=out_chan) if not is_last else nn.Identity(),
                    ]
                )
            )

        bottleneck_capacity = dims[-1]
        self.mid_block1 = block_klass(bottleneck_capacity, bottleneck_capacity, time_embedding_dim=time_dim)
        self.mid_attn = Residual(PreNorm(bottleneck_capacity, Attention(bottleneck_capacity)))
        self.mid_block2 = block_klass(bottleneck_capacity, bottleneck_capacity, time_embedding_dim=time_dim)

        

        for idx, (in_chan, out_chan) in enumerate(reversed(resolution_translations[1:])):
            is_last = idx >= (num_resolutions - 1)

            self.decoder.append(
                nn.ModuleList(
                    [
                        block_klass(in_channels=out_chan * 2, out_channels=in_chan, time_embedding_dim=time_dim),
                        block_klass(in_channels=in_chan, out_channels=in_chan, time_embedding_dim=time_dim),
                        Residual(fn=PreNorm(num_channels=in_chan, fn=LinearAttention(num_channels=in_chan))),
                        Upsample(num_channels=in_chan) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_chan = default(out_channels, in_channels)
        self.final_conv = nn.Sequential(
            block_klass(in_channels=n_features, out_channels=n_features), 
            nn.Conv2d(in_channels=n_features, out_channels=out_chan, kernel_size=1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_projection(time) if exists(self.time_projection) else None

        noisy_latent_representation_stack = []

        # downsample
        for block1, block2, attn, downsample in self.encoder:
            x = block1(x, time_emb=t)
            x = block2(x, time_emb=t)
            x = attn(x)
            noisy_latent_representation_stack.append(x)
            x = downsample(x)
        
        # bottleneck
        x = self.mid_block1(x, time_emb=t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=t)

        # upsample
        for block1, block2, attn, upsample in self.decoder:
            x = torch.cat((x, noisy_latent_representation_stack.pop()), dim=1)
            x = block1(x, time_emb=t)
            x = block2(x, time_emb=t)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x)

def plot_samples(samples):
    '''
    Plot samples
    :param samples: samples to plot
    '''
    n_rows = int(np.sqrt(samples.shape[0]))
    n_cols = n_rows
    samples = np.transpose(samples, (0, 2, 3, 1))
    samples = samples * 0.5 + 0.5
    samples = np.clip(samples, 0, 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if samples.shape[-1] == 1:
            ax.imshow(samples[i].squeeze(), cmap='gray')
        else:
            ax.imshow(samples[i])
        ax.axis('off')
    plt.show()

class VanillaDDPM(nn.Module):
    def __init__(self, args, image_size, channels, with_time_emb=True):
        '''
        VanillaDDPM module
        :param args: arguments
        :param image_size: size of the image
        :param in_channels: number of input channels
        :param with_time_emb: whether to use time embeddings
        '''
        super().__init__()
        self.reverse_transform = Compose([
                                        Lambda(lambda t: (t + 1) / 2),
                                        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
                                        Lambda(lambda t: t * 255.),
                                        Lambda(lambda t: t.numpy().astype(np.uint8)),
                                        ToPILImage(),
                                    ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.denoising_model = UNet(args.n_features, args.init_channels, channels, args.channel_scale_factors, channels, with_time_emb, args.resnet_block_groups, args.use_convnext, args.convnext_scale_factor).to(self.device)
        self.scheduler = LinearScheduler(args.beta_start, args.beta_end, args.timesteps)
        self.forward_diffusion_model = ForwardDiffusion(self.scheduler.sqrt_alphas_cumprod, self.scheduler.sqrt_one_minus_alphas_cumprod, self.reverse_transform)
        self.sampler = Sampler(self.scheduler.betas, args.timesteps, args.sample_timesteps, args.ddpm)
        self.optimizer = torch.optim.Adam(self.denoising_model.parameters(), lr = args.lr)
        self.criterion = get_loss
        self.n_epochs = args.n_epochs
        self.timesteps = args.timesteps
        self.sample_and_save_freq = args.sample_and_save_freq
        self.loss_type = args.loss_type
        self.image_size = image_size
        self.num_channels = channels
        self.dataset = args.dataset
        self.no_wandb = args.no_wandb


    def train_model(self, dataloader):
        '''
        Train the model
        :param dataloader: dataloader
        '''
        best_loss = np.inf
        create_checkpoint_dir()
        for epoch in tqdm(range(self.n_epochs), desc='Training DDPM', leave=True):
            acc_loss = 0.0
            with tqdm(dataloader, desc=f'Batches', leave=False) as pbar:
                for step,batch in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    batch_size = batch[0].shape[0]
                    batch = batch[0].to(self.device)
                    t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                    loss = self.criterion(forward_diffusion_model=self.forward_diffusion_model, denoising_model=self.denoising_model, x_start=batch, t=t, loss_type=self.loss_type)
                    loss.backward()
                    self.optimizer.step()
                    acc_loss += loss.item() * batch_size

                    pbar.set_postfix(Epoch=f"{epoch+1}/{self.n_epochs}", Loss=f"{loss:.4f}")
                    pbar.update()

                    # save generated images
                if epoch % self.sample_and_save_freq == 0:
                    samples = self.sampler.sample(model=self.denoising_model, image_size=self.image_size, batch_size=16, channels=self.num_channels)
                    all_images = samples[-1] 
                    all_images = (all_images + 1) * 0.5
                    # all_images is a numpy array, plot 9 images from it
                    fig = plt.figure(figsize=(10, 10))
                    n_row = np.sqrt(all_images.shape[0]).astype(int)
                    n_col = n_row
                    # use subplots
                    for i in range(n_row*n_col):
                        plt.subplot(n_col, n_row, i+1)
                        if self.num_channels == 1:
                            plt.imshow(all_images[i].squeeze(), cmap='gray')
                        else:
                            plt.imshow(all_images[i].transpose(1,2,0))
                        plt.axis('off')
                    #save figure wandb
                    if not self.no_wandb:
                        wandb.log({"DDPM Samples": fig})
                    plt.close(fig)

            if acc_loss/len(dataloader.dataset) < best_loss:
                best_loss = acc_loss/len(dataloader.dataset)
                torch.save(self.denoising_model.state_dict(), os.path.join(models_dir,'VanillaDDPM',f'VanDDPM_{self.dataset}.pt'))
            if not self.no_wandb:
                wandb.log({"DDPM Loss": acc_loss/len(dataloader.dataset)})
    
    def outlier_score(self, x_start, t):
        '''
        Compute the outlier score
        :param x_start: input image
        :param t: time
        '''
        noise = torch.randn_like(x_start)

        x_noisy = self.forward_diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoising_model(x_noisy, t)

        if self.loss_type == 'l1':
            loss = nn.L1Loss(reduction = 'none')
            elementwise_loss = torch.mean(loss(noise, predicted_noise).reshape(x_start.shape), dim=(1,2,3))
        elif self.loss_type == 'l2':
            loss = nn.MSELoss(reduction = 'none')
            elementwise_loss = torch.mean(loss(noise, predicted_noise).reshape(x_start.shape), dim=(1,2,3))
        elif self.loss_type == "huber":
            loss = nn.HuberLoss(reduction = 'none')
            elementwise_loss = torch.mean(loss(noise, predicted_noise).reshape(x_start.shape), dim=(1,2,3))
        else:
            raise NotImplementedError()

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
        self.denoising_model.eval()
        val_loss = 0.0
        val_scores = []
        for step, batch in enumerate(val_loader):
            batch_size = batch[0].shape[0]
            batch = batch[0].to(self.device)
            t = torch.ones((batch_size,), device=self.device).long() * 0
            score = outlier_score(forward_diffusion_model=self.forward_diffusion_model, denoising_model=self.denoising_model, x_start=batch, t=t, loss_type=self.loss_type)
            val_scores.append(score.cpu().numpy())
        val_scores = np.concatenate(val_scores)

        out_scores = []

        out_scores = []
        for step, batch in enumerate(out_loader):
            batch_size = batch[0].shape[0]
            batch = batch[0].to(self.device)
            t = torch.ones((batch_size,), device=self.device).long() * 0
            out_scores.append(outlier_score(forward_diffusion_model=self.forward_diffusion_model, denoising_model=self.denoising_model, x_start=batch, t=t, loss_type=self.loss_type).cpu().numpy())
        out_scores = np.concatenate(out_scores)
        
        y_true = np.concatenate([np.zeros_like(val_scores), np.ones_like(out_scores)], axis=0)
        y_score = np.concatenate([val_scores, out_scores], axis=0)
        auc_score = roc_auc_score(y_true, y_score)
        if auc_score < 0.2:
            auc_score = 1. - auc_score
        print('AUC score: {:.5f}'.format(auc_score))

        plt.hist(val_scores, bins=100, alpha=0.5, label='In')
        plt.hist(out_scores, bins=100, alpha=0.5, label='Out')
        plt.legend(loc='upper right')
        plt.title('{} vs {} AUC: {:.4f}'.format(in_name, out_name, auc_score))
        plt.show()
    
    @torch.no_grad()
    def sample(self, batch_size=16):
        '''
        Sample images
        :param batch_size: batch size
        '''
        samps = self.sampler.sample(model=self.denoising_model, image_size=self.image_size, batch_size=batch_size, channels=self.num_channels)[-1]
        plot_samples(samps)

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
    def __init__(self, betas, timesteps=1000, sample_timesteps=100, ddpm=1.0):
        '''
        Sampler module
        :param betas: beta values
        :param timesteps: number of timesteps
        :param sample_timesteps: number of sample timesteps
        :param ddpm: diffusion coefficient
        '''
        self.betas = betas
        self.alphas = (1-self.betas).cumprod(dim=0)
        self.timesteps = timesteps
        self.sample_timesteps = sample_timesteps
        self.ddpm = ddpm
        self.scaling = timesteps//sample_timesteps
    
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

        if tau_index == 0:
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
        
        for i in tqdm(range(self.sample_timesteps-1,-1,-1), desc="Sampling", leave=False):
            scaled_i = i*self.scaling
            img = self.p_sample(model, img, torch.full((b,), scaled_i, device=device, dtype=torch.long), i)
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

    x_noisy = forward_diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)
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

def outlier_score(forward_diffusion_model, denoising_model, x_start, t, loss_type):
    '''
    Compute the outlier score
    :param forward_diffusion_model: forward diffusion model
    :param denoising_model: denoising model
    :param x_start: input image
    :param t: time
    :param loss_type: type of loss
    '''
    noise = torch.randn_like(x_start)

    x_noisy = forward_diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoising_model(x_noisy, t)

    if loss_type == 'l1':
        loss = nn.L1Loss(reduction = 'none')
        elementwise_loss = torch.mean(loss(noise, predicted_noise).reshape(x_start.shape), dim=(1,2,3))
    elif loss_type == 'l2':
        loss = nn.MSELoss(reduction = 'none')
        elementwise_loss = torch.mean(loss(noise, predicted_noise).reshape(x_start.shape), dim=(1,2,3))
    elif loss_type == "huber":
        loss = nn.HuberLoss(reduction = 'none')
        elementwise_loss = torch.mean(loss(noise, predicted_noise).reshape(x_start.shape), dim=(1,2,3))
    else:
        raise NotImplementedError()

    return elementwise_loss