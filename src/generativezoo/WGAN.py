from models.GAN.WGAN import *
import torch
from data.Dataloaders import *
import wandb

dataloader, _, _ = tinyimagenet_train_loader(batch_size=512, input_shape=64, normalize=False)

wandb.init(project="WGAN",
           config={
               "latent_dim": 1024,
               "d": 64,
               "channels": 3,
               "epochs": 100,
               "batch_size": 256,
               "lr": 0.0002,
               "beta1": 0.5,
               "beta2": 0.999,
               "n_critic": 5,
               "dataset": "tinyimagenet"
           },
           name="WGAN_tinyimagenet")

model = WGAN(latent_dim=1024, d=64, channels = 3, imgSize=64, n_epochs=400, batch_size=512, dataset="tinyimagenet")
model.train_model(dataloader)