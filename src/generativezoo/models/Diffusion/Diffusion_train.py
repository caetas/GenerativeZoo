
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from Diffusion import *
from torchvision.transforms import Compose, Lambda, ToPILImage
from torchvision.utils import make_grid
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def mnist_train_loader(batch_size):
    training_data = datasets.MNIST(root='./../../../../data/raw', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def mnist_val_loader(batch_size):
    validation_data = datasets.MNIST(root='./../../../../data/raw', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return validation_loader

beta_start = 0.0001
beta_end = 0.02
timesteps = 300
image_size = 28
num_channels = 1
batch_size = 128
results_folder_name = './../../../../reports'
sample_and_save_freq = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
learninig_rate = 1e-3
epochs = 50
results_folder = Path(results_folder_name)

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
sampler = Sampler(betas=scheduler.betas, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, sqrt_one_by_alphas=scheduler.sqrt_one_by_alphas, posterior_variance=scheduler.posterior_variance, timesteps=timesteps)

model = DDPM(n_features=image_size, in_channels=num_channels, channel_scale_factors=(1, 2, 4,)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learninig_rate)
criterion = get_loss

train(image_size=image_size, num_channels=num_channels, epochs=epochs, timesteps=timesteps, sample_and_save_freq=sample_and_save_freq, save_folder=results_folder, forward_diffusion_model=forward_diffusion, denoising_model=model, criterion=criterion, optimizer=optimizer, dataloader=dataloader, sampler=sampler, device=device)