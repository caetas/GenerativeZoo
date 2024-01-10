import numpy as np
import torch
from models.VAE.VanillaVAE import VanillaVAE
from data.Dataloaders import cifar_train_loader, cifar_val_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = cifar_train_loader(256)
val_loader = cifar_val_loader(32)
print(device)

# mock data for testing
x = torch.randn(1, 3, 32, 32).to(device)
# Create the model
model = VanillaVAE(input_shape = 32, input_channels = 3, latent_dim = 128, hidden_dims=[32,64,128,256]).to(device)

model.train()
model.train_model(train_loader, epochs=100, device=device)