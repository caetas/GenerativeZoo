import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os
from config import figures_dir, models_dir

class VanillaVAE(nn.Module):
    def __init__(self, input_shape, input_channels, latent_dim, hidden_dims = None, lr = 5e-3, batch_size = 64):
        super(VanillaVAE, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.final_channels = input_channels
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        # each layer decreases the h and w by 2, so we need to divide by 2**(number of layers) to know the factor for the flattened input
        self.multiplier = int(self.input_shape/(2**len(hidden_dims)))
        self.last_channel = hidden_dims[-1]
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, h_dim, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            input_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*(self.multiplier**2), latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*(self.multiplier**2), latent_dim)

        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*(self.multiplier**2))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride = 2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.final_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)

        # Split into mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def load_state_dictionary(self, path):
        self.load_state_dict(torch.load(path))
        return self

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, self.last_channel, self.multiplier, self.multiplier)
        z = self.decoder(z)
        return self.final_layer(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, z):
        return self.decode(z)
    
    def loss_function(self, recon_x, x, mu, logvar, al = 0.0):
        loss_mse = nn.MSELoss()
        mse = loss_mse(x, recon_x)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim=0)
        return mse + kld/(self.batch_size**2)
    
    def adversarial_loss(self, x, y):
        loss = nn.BCEWithLogitsLoss()
        return loss(x, y)
    
    def create_grid(self, device, figsize=(10, 10), title=None):
        samples = self.generate(torch.randn(9, self.latent_dim).to(device)).detach().cpu()
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=figsize)
        grid_size = int(np.sqrt(samples.shape[0]))
        grid = torchvision.utils.make_grid(samples, nrow=grid_size).permute(1, 2, 0)
        # save grid image
        plt.imshow(grid)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.savefig(os.path.join(figures_dir, f"AdvVAE_{title}.png"))
        plt.close(fig)

class Discriminator(nn.Module):
    def __init__(self, input_shape, input_channels, hidden_dims = None, lr = 5e-3, batch_size = 64):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.final_channels = input_channels
        self.lr = lr
        self.batch_size = batch_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        # each layer decreases the h and w by 2, so we need to divide by 2**(number of layers) to know the factor for the flattened input
        self.multiplier = int(self.input_shape/(2**len(hidden_dims)))
        self.last_channel = hidden_dims[-1]
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, h_dim, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            input_channels = h_dim

        modules.append(nn.Flatten())
        modules.append(nn.Linear(hidden_dims[-1]*(self.multiplier**2), 1))
        modules.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*(self.multiplier**2), 1)

    def forward(self, x):
        x = self.encoder(x)
        return x

    def loss_function(self, x, y):
        loss = nn.BCEWithLogitsLoss()
        return loss(x, y)
    
class AdversarialAE(nn.Module):
    def __init__(self, input_shape, input_channels, latent_dim, hidden_dims = None, lr = 5e-3, batch_size = 64):
        super(AdversarialAE, self).__init__()
        self.vae = VanillaVAE(input_shape, input_channels, latent_dim, hidden_dims, lr, batch_size)
        self.discriminator = Discriminator(input_shape, input_channels, hidden_dims, lr, batch_size)
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, x):
        image,_,_ = self.vae(x)
        label = self.discriminator(image)
        return image, label
    
    def create_grid(self, device, figsize=(10, 10), title=None):
        samples = self.vae.generate(torch.randn(9, self.vae.latent_dim).to(device)).detach().cpu()
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=figsize)
        grid_size = int(np.sqrt(samples.shape[0]))
        grid = torchvision.utils.make_grid(samples, nrow=grid_size).permute(1, 2, 0)
        # save grid image
        plt.imshow(grid)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.savefig(os.path.join(figures_dir, f"AdvAE_{title}.png"))
        plt.close(fig)

    def create_validation_grid(self, data_loader, device, figsize=(10, 10), title=None):
        # get a batch of data
        x, _ = next(iter(data_loader))[:10]
        x = x.to(device)
        # get reconstruction
        with torch.no_grad():
            recon_x,_,_ = self.vae(x)
        # prep images
        x = x.detach().cpu()
        recon_x = recon_x.detach().cpu()
        x = (x + 1) / 2
        recon_x = (recon_x + 1) / 2
        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
        # input images on top row, reconstructions on bottom
        for images, row in zip([x, recon_x], axes):
            for img, ax in zip(images, row):
                ax.imshow(img.permute(1, 2, 0))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if title:
            plt.title(title)
        plt.savefig(os.path.join(figures_dir, f"AdvAE_val_{title}.png"))

    def train_model(self, data_loader, val_loader, epochs, device):

        