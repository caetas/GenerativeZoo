
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
    save_folder.mkdir(exist_ok = True)
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
                    plt.savefig(save_folder / f"Epoch_{epoch}_Step_{step}.png")
                    plt.close(fig)
        if acc_loss/len(dataloader.dataset) < best_loss:
            best_loss = acc_loss/len(dataloader.dataset)
            torch.save(denoising_model.state_dict(), save_folder / f"DDPM.pt")

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