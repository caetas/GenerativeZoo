import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os
from config import figures_dir, models_dir

class ConditionalVAE(nn.Module):
    def __init__(self, input_shape, input_channels, latent_dim, num_classes, hidden_dims = None, lr = 1e-3):
        super(ConditionalVAE, self).__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.final_channels = input_channels
        self.latent_dim = latent_dim
        self.lr = lr

        self.embed_class = nn.Linear(num_classes, self.input_shape**2)
        self.embed_data = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        # each layer decreases the h and w by 2, so we need to divide by 2**(number of layers) to know the factor for the flattened input
        self.multiplier = int(self.input_shape/(2**len(hidden_dims)))
        self.last_channel = hidden_dims[-1]
        modules = []

        # to account for classes
        input_channels += 1

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

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1]*(self.multiplier**2))

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
        return self.fc_mu(x), self.fc_logvar(x)
    
    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, self.last_channel, self.multiplier, self.multiplier)
        z = self.decoder(z)
        return self.final_layer(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + (eps*std)
    
    def forward(self, x, y):
        x = self.embed_data(x)
        y_emb = self.embed_class(y)
        y_emb = y_emb.view(-1, self.input_channels, self.input_shape, self.input_shape)
        x = torch.cat([x, y_emb], dim = 1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, y], dim = 1)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        loss_mse = nn.MSELoss()
        mse = loss_mse(x, recon_x)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim=0)
        return mse + kld*0.00005
    
    def generate(self, z, y):
        z = torch.cat([z, y], dim = 1)
        return self.decode(z)
    
    def one_hot_encode(self, y):
        y_onehot = torch.zeros(y.size(0), self.num_classes)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        return y_onehot
    
    def create_grid(self, device, figsize=(10, 10), title=None):
        n = 9
        z = torch.randn(n, self.latent_dim).to(device)
        y = torch.eye(self.num_classes).to(device)
        y = y[:n]
        samples = self.generate(z,y).detach().cpu()
        samples = (samples + 1) / 2
        fig = plt.figure(figsize=figsize)
        grid_size = int(np.sqrt(samples.shape[0]))
        grid = torchvision.utils.make_grid(samples, nrow=grid_size).permute(1, 2, 0)
        # save grid image
        plt.imshow(grid)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.savefig(os.path.join(figures_dir, f"CVAE_{title}.png"))
        plt.close(fig)
        return grid
    
    def train_model(self, train_loader, epochs, device):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        self.train()
        epochs_bar = trange(epochs)
        for epoch in epochs_bar:
            acc_loss = 0
            for x, y in train_loader:
                x = x.to(device)
                y = self.one_hot_encode(y).float().to(device)
                recon_x, mu, logvar = self(x, y)
                loss = self.loss_function(recon_x, x, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc_loss += loss.item()
            epochs_bar.set_description(f"Loss: {acc_loss/len(train_loader.dataset):.8f}")
            epochs_bar.refresh()
            if epoch % 5 == 0:
                self.create_grid(device, title=f"Epoch_{epoch}")
        