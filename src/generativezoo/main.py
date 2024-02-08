import numpy as np
import torch
from models.VAE.ConditionalVAE import ConditionalVAE
from models.VAE.VQVAE import VQVAE
from models.GANs.AdversarialAE import AdversarialAE
from data.Dataloaders import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
train_loader = svhn_train_loader(batch_size=batch_size, normalize=True)
#train_loader = mnist_train_loader(batch_size=batch_size, input_shape=32, normalize=True)
val_loader = svhn_val_loader(batch_size=batch_size, normalize=True)
#val_loader = mnist_val_loader(batch_size=batch_size, input_shape=32, normalize=True)
print(device)

# mock data for testing
#x = torch.randn(1, 3, 32, 32).to(device)
model = AdversarialAE(input_shape = 32, input_channels = 3, latent_dim = 128, hidden_dims=[32,64,128,256], batch_size=batch_size).to(device)
model.train_model(train_loader, val_loader, epochs=100, device=device)
# Create the model
#model = VQVAE(input_shape = 32, input_channels = 3, embedding_dim = 64, num_embeddings = 512, batch_size=batch_size).to(device)
#model = model.load_pretrained_weights('./../../models/VQVAE.pt')
#model.create_histogram(train_loader, device)
#model.create_grid(train_loader, device)
#model.train_model(train_loader, epochs=100, device=device)
#model = ConditionalVAE(input_shape = 32, input_channels = 3, latent_dim = 128, hidden_dims=[16,32,64,128], batch_size=batch_size, num_classes=10).to(device)
#model = ConditionalVAE(input_shape = 28, input_channels = 1, latent_dim = 64, num_classes=10, hidden_dims=[64, 128]).to(device)
#model.train()
#model.train_model(train_loader, epochs=100, device=device)