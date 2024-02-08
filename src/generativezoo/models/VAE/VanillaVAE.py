##############################################################################################################
########### Code based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py ############
##############################################################################################################

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os
from config import figures_dir, models_dir
import wandb

class VanillaVAE(nn.Module):
    def __init__(self, input_shape, input_channels, latent_dim, device, hidden_dims = None, lr = 5e-3, batch_size = 64):
        '''Vanilla VAE model
        Args:
        input_shape: int, input shape of the image
        input_channels: int, number of channels of the input image
        latent_dim: int, dimension of the latent space
        device: torch.device, device to run the model
        hidden_dims: list, list of integers with the number of channels of the hidden layers, if none it will be [32, 64, 128, 256, 512]
        lr: float, learning rate for the optimizer
        batch_size: int, batch size for the training
        '''
        super(VanillaVAE, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.final_channels = input_channels
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

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

        self.encoder.to(self.device)
        self.fc_mu.to(self.device)
        self.fc_logvar.to(self.device)
        self.decoder_input.to(self.device)
        self.decoder.to(self.device)
        self.final_layer.to(self.device)

    def encode(self, x):
        '''Encode the input into the latent space
        Args:
        x: torch.Tensor, input tensor
        Returns:
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        '''
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)

        # Split into mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        '''Decode the latent space into the input space
        Args:
        z: torch.Tensor, latent space tensor
        Returns:
        x: torch.Tensor, reconstructed input tensor
        '''
        z = self.decoder_input(z)
        z = z.view(-1, self.last_channel, self.multiplier, self.multiplier)
        z = self.decoder(z)
        return self.final_layer(z)

    def reparameterize(self, mu, logvar):
        '''Reparameterize the latent space
        Args:
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        Returns:
        z: torch.Tensor, reparameterized latent space
        '''
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        '''Forward pass of the model
        Args:
        x: torch.Tensor, input tensor
        Returns:
        x: torch.Tensor, reconstructed input tensor
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        '''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, z):
        '''Generate samples from the latent space
        Args:
        z: torch.Tensor, latent space tensor
        Returns:
        x: torch.Tensor, reconstructed input tensor
        '''
        return self.decode(z)
    
    def loss_function(self, recon_x, x, mu, logvar):
        '''Loss function of the model
        Args:
        recon_x: torch.Tensor, reconstructed input tensor
        x: torch.Tensor, input tensor
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        Returns:
        loss: torch.Tensor, loss of the model
        '''
        loss_mse = nn.MSELoss()
        mse = loss_mse(x, recon_x)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim=0)
        return mse + kld/(self.batch_size**2)
    
    def create_grid(self, figsize=(10, 10), title=None, train = False):
        '''Create a grid of samples from the latent space
        Args:
        figsize: tuple, size of the figure
        title: str, title of the figure
        train: bool, if the model is training or not
        '''
        samples = self.generate(torch.randn(9, self.latent_dim).to(self.device)).detach().cpu()
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=figsize)
        grid_size = int(np.sqrt(samples.shape[0]))
        grid = torchvision.utils.make_grid(samples, nrow=grid_size).permute(1, 2, 0)
        # save grid image
        plt.imshow(grid)
        plt.axis('off')
        if title:
            plt.title(title)
        if train:
            wandb.log({title: fig})
        else:
            plt.savefig(os.path.join(figures_dir, f"VAE_{title}.png"))
        plt.close(fig)
    
    def train_model(self, data_loader, epochs):
        '''Train the model
        Args:
        data_loader: torch.utils.data.DataLoader, data loader for the training
        epochs: int, number of epochs to train the model
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        epochs_bar = trange(epochs, desc = "Epochs", leave = True)
        for epoch in epochs_bar:
            acc_loss = 0.0
            for _,(data,_) in enumerate(data_loader):
                x = data.to(self.device)
                recon_x, mu, logvar = self(x)
                loss = self.loss_function(recon_x, x, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc_loss += loss.item()
            #write below the bar
            epochs_bar.set_description("Loss: {:.8f}".format(acc_loss/len(data_loader.dataset)))
            epochs_bar.refresh()
            wandb.log({"loss": acc_loss/len(data_loader.dataset)})
            if epoch % 5 == 0:
                self.create_grid(title=f"Epoch_{epoch}", train = True)
        torch.save(self.state_dict(), os.path.join(models_dir, f"VAE_{epoch}.pt"))
    
    def eval_model(self, data_loader):
        '''Evaluate the model
        Args:
        data_loader: torch.utils.data.DataLoader, data loader for the evaluation
        '''
        self.eval()
        with torch.no_grad():
            acc_loss = 0.0
            for _,(data,_) in enumerate(data_loader):
                x = data.to(self.device)
                recon_x, mu, logvar = self(x)
                loss = self.loss_function(recon_x, x, mu, logvar)
                acc_loss += loss.item()
            print("Loss: {:.4f}".format(acc_loss/len(data_loader.dataset)))

    