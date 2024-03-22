from models.GAN.WGAN import *
import torch
from data.Dataloaders import *
import wandb

dataloader, _, _ = cifar_train_loader(batch_size=64, input_shape=32, normalize=False)

model = WGAN(latent_dim=100, d=64, channels = 3)
model.train_model(dataloader)