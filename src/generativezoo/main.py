import numpy as np
import torch
from models.VAE.ConditionalVAE import ConditionalVAE
from data.Dataloaders import cifar_train_loader, cifar_val_loader, mnist_train_loader, mnist_val_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = mnist_train_loader(256)
print(device)

# mock data for testing
x = torch.randn(1, 1, 28, 28).to(device)
# y should be a float
y = torch.tensor([0, 1, 0, 0, 0, 0 ,0, 0, 0, 0]).float().to(device)
# unsqueeze to add a batch dimension
y = y.unsqueeze(0)
# Create the model
#model = VanillaVAE(input_shape = 32, input_channels = 3, latent_dim = 128, hidden_dims=[16,32,64,128]).to(device)
model = ConditionalVAE(input_shape = 28, input_channels = 1, latent_dim = 32, num_classes=10, hidden_dims=[32, 64]).to(device)
model.train()
model.train_model(train_loader, epochs=50, device=device)