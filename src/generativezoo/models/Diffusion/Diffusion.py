import torch.nn as nn
from einops import rearrange
from torch import einsum
import torch
import math
from functools import partial
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

######## Code from https://github.com/DhruvSrikanth/DenoisingDiffusionProbabilisticModels/tree/master ##########

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Attention(nn.Module):
    def __init__(self, num_channels, num_heads=4, head_dim=32):
        super().__init__()
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads
        self.to_qkv = nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=1)
        
    def forward(self, x):
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
        super().__init__()
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(num_gruops=groups, num_channels=out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.projection(x)
        x = self.group_norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_embedding_dim=None, channel_scale_factor=2, normalize=True):
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

class DDPM(nn.Module):
    def __init__(self, n_features, init_channels=None, out_channels=None, channel_scale_factors=(1, 2, 4, 8), in_channels=3, with_time_emb=True, resnet_block_groups=8, use_convnext=True, convnext_scale_factor=2):
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
    
    def load_pretrained_weights(self, pretrained_weights_path):
        pretrained_state_dict = torch.load(pretrained_weights_path)
        self.load_state_dict(pretrained_state_dict, strict=False)

class LinearScheduler():
    def __init__(self, beta_start=0.0001, beta_end=0.02, timesteps=1000):
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
    def __init__(self, betas, timesteps, ddpm = 1.0):
        self.betas = betas
        self.alphas = (1-self.betas).cumprod(dim=0)
        self.timesteps = timesteps
        self.ddpm = ddpm
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = extract_time_index(self.betas, t, x.shape)
        alpha_t = extract_time_index(self.alphas, t, x.shape)
        x0_t = (x - (1-alpha_t).sqrt()*model(x, t))/alpha_t.sqrt()

        if t_index == 0:
            return x0_t
        else:
            alpha_prev_t = extract_time_index(self.alphas, t-1, x.shape)
            c1 = self.ddpm*((1 - alpha_t/alpha_prev_t) * (1-alpha_prev_t) / (1 - alpha_t)).sqrt()
            c2  = ((1-alpha_prev_t) - c1**2).sqrt()
            noise = torch.randn_like(x)
            return x0_t*alpha_prev_t.sqrt() + c2*model(x,t) +  c1* noise

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    
class Accelerated_Sampler():
    def __init__(self, betas, timesteps=1000, reduced_timesteps=100, ddpm=1.0):
        self.betas = betas
        self.alphas = (1-self.betas).cumprod(dim=0)
        self.timesteps = timesteps
        self.reduced_timesteps = reduced_timesteps
        self.ddpm = ddpm
        self.scaling = timesteps//reduced_timesteps
    
    @torch.no_grad()
    def p_sample(self, model, x, t, tau_index):
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
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in reversed(range(0, self.reduced_timesteps)):
            scaled_i = i*self.scaling
            img = self.p_sample(model, img, torch.full((b,), scaled_i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def get_loss(forward_diffusion_model, denoising_model, x_start, t, noise=None, loss_type="l2"):
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

def train(image_size, num_channels, epochs, timesteps, sample_and_save_freq, save_folder, forward_diffusion_model, denoising_model, criterion, optimizer, dataloader, sampler, device):
    best_loss = np.inf
    loss_type="huber"
    for epoch in range(epochs):
        acc_loss = 0.0
        with tqdm(dataloader, desc=f'Training DDPM') as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                batch_size = batch[0].shape[0]
                batch = batch[0].to(device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, timesteps, (batch_size,), device=device).long()
                loss = criterion(forward_diffusion_model=forward_diffusion_model, denoising_model=denoising_model, x_start=batch, t=t, loss_type=loss_type)
                # if step % 100 == 0:
                #     print(f"Epoch {epoch} Loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                acc_loss += loss.item() * batch_size

                pbar.set_postfix(Epoch=f"{epoch+1}/{epochs}", Loss=f"{loss:.4f}")
                pbar.update()


                # save generated images
                if step != 0 and step % sample_and_save_freq == 0:
                    samples = sampler.sample(model=denoising_model, image_size=image_size, batch_size=16, channels=num_channels)
                    all_images = samples[-1] 
                    all_images = (all_images + 1) * 0.5
                    # all_images is a numpy array, plot 9 images from it
                    fig = plt.figure(figsize=(10, 10))
                    # use subplots
                    for i in range(9):
                        plt.subplot(3, 3, i+1)
                        plt.imshow(all_images[i].squeeze(), cmap='gray')
                        plt.axis('off')
                    plt.savefig(os.path.join(save_folder,f"Epoch_{epoch}_Step_{step}.png"))
                    plt.close(fig)
        if acc_loss/len(dataloader.dataset) < best_loss:
            best_loss = acc_loss/len(dataloader.dataset)
            torch.save(denoising_model.state_dict(), os.path.join(save_folder,f"DDPM.pt"))
    
