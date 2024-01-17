from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Compose, Lambda, ToPILImage
from torchvision.utils import make_grid
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data.Dataloaders import mnist_train_loader, mnist_val_loader
from models.Diffusion.Diffusion import *
from config import data_raw_dir, models_dir

beta_start = 0.0001
beta_end = 0.02
timesteps = 300
image_size = 28
num_channels = 1
batch_size = 128
results_folder = data_raw_dir
sample_and_save_freq = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
learninig_rate = 1e-3
epochs = 50

scheduler = LinearScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

forward_diffusion = ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, reverse_transform=reverse_transform)


dataloader = mnist_train_loader(batch_size=batch_size)
sampler = Sampler(betas=scheduler.betas, timesteps=timesteps, ddpm=0.0)

model = DDPM(n_features=image_size, in_channels=num_channels, channel_scale_factors=(1, 2, 4,)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learninig_rate)
criterion = get_loss

train(image_size=image_size, num_channels=num_channels, epochs=epochs, timesteps=timesteps, sample_and_save_freq=sample_and_save_freq, save_folder=results_folder, forward_diffusion_model=forward_diffusion, denoising_model=model, criterion=criterion, optimizer=optimizer, dataloader=dataloader, sampler=sampler, device=device)