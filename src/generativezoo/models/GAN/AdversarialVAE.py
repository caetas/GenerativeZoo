##############################################################################################################
########### Code based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py ############
##############################################################################################################

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os
from config import figures_dir, models_dir
import wandb

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'AdversarialVAE')):
    os.makedirs(os.path.join(models_dir, 'AdversarialVAE'))

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
        self.multiplier = np.round(self.input_shape/(2**len(hidden_dims)), 0).astype(int)
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
    
    def loss_function(self, recon_x, x, mu, logvar):
        loss_mse = nn.MSELoss()
        mse = loss_mse(x, recon_x)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim=0)
        return mse + kld/(self.batch_size**2)
    

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
        self.multiplier = np.round(self.input_shape/(2**len(hidden_dims)), 0).astype(int)
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
    
class AdversarialVAE(nn.Module):
    def __init__(self, input_shape, device, input_channels, latent_dim, n_epochs, hidden_dims = None, lr = 5e-3, batch_size = 64, gen_weight = 0.002, recon_weight = 0.002, sample_and_save_frequency = 10, dataset = 'mnist'):
        super(AdversarialVAE, self).__init__()
        self.device = device
        self.vae = VanillaVAE(input_shape, input_channels, latent_dim, hidden_dims, lr, batch_size).to(self.device)
        self.discriminator = Discriminator(input_shape, input_channels, hidden_dims, lr, batch_size).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.gen_weight = gen_weight
        self.recon_weight = recon_weight
        self.n_epochs = n_epochs
        self.sample_and_save_frequency = sample_and_save_frequency
        self.dataset = dataset

    def forward(self, x):
        image,_,_ = self.vae(x)
        label = self.discriminator(image)
        return image, label
    
    def create_grid(self, figsize=(10, 10), title=None, train = False):
        samples = self.vae.generate(torch.randn(9, self.vae.latent_dim).to(self.device)).detach().cpu()
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
            wandb.log({f"Samples": fig})
        else:
            plt.show()
        plt.close(fig)

    def create_validation_grid(self, data_loader, figsize=(10, 4), title=None, train = False):
        # get a batch of data
        x, _ = next(iter(data_loader))
        x = x.to(self.device)
        x = x[:10]
        # get reconstruction
        with torch.no_grad():
            recon_x,_,_ = self.vae(x)
        x = (x.detach().cpu() + 1) / 2
        recon_x = (recon_x.detach().cpu() + 1) / 2
        fig = plt.figure(figsize=figsize)
        samples = torch.cat((x, recon_x), 0)
        grid = torchvision.utils.make_grid(samples, nrow=x.shape[0]).permute(1, 2, 0)
        # save grid image
        plt.imshow(grid)
        plt.axis('off')
        if title:
            plt.title(title)
        if train:
            wandb.log({f"Reconstruction": fig})
        else:
            plt.show()
        plt.close(fig)
        

    def train_model(self, data_loader, val_loader):

        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Optimizers
        optimizer_VAE = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        best_loss = np.inf

        epochs_bar = trange(self.n_epochs, desc="Loss: ------", leave=True)

        create_checkpoint_dir()
        # ----------
        #  Training
        # ----------
        for epoch in epochs_bar:
            
            acc_g_loss = 0.0
            acc_d_loss = 0.0

            for (imgs, _) in tqdm(data_loader, desc = 'Batches', leave=False):

                # Adversarial ground truths
                valid = torch.ones(imgs.size(0), 1).to(self.device)
                fake = torch.zeros(imgs.size(0), 1).to(self.device)

                # Configure input
                real_imgs = imgs.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_VAE.zero_grad()

                # Generate a batch of images
                recon_imgs, mu, logvar = self.vae(real_imgs)
                noise = torch.randn(imgs.size(0), self.vae.latent_dim).to(self.device)
                gen_imgs = self.vae.decode(noise)

                # Loss measures generator's ability to fool the self.discriminator
                validity_recon = self.discriminator(recon_imgs)
                validity_gen = self.discriminator(gen_imgs)
                g_loss = self.recon_weight*adversarial_loss(validity_recon, valid) + self.gen_weight*adversarial_loss(validity_gen, valid) + self.vae.loss_function(recon_imgs, real_imgs, mu, logvar)
                acc_g_loss += g_loss.item()*imgs.size(0)

                g_loss.backward()
                optimizer_VAE.step()

                epochs_bar.set_description(f"Loss: {g_loss.item():.5f}")

                # ---------------------
                #  Train self.discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = self.discriminator(real_imgs)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = self.discriminator(gen_imgs.detach())
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total self.discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                acc_d_loss += d_loss.item()*imgs.size(0)

                d_loss.backward()
                optimizer_D.step()

            epochs_bar.set_description(f"Loss: {acc_g_loss/len(data_loader.dataset):.4f} - D Loss: {acc_d_loss/len(data_loader.dataset):.4f}")
            wandb.log({"Generator Loss": acc_g_loss/len(data_loader.dataset), "Discriminator Loss": acc_d_loss/len(data_loader.dataset)})


            if (epoch+1) % self.sample_and_save_frequency == 0 or epoch == 0:
                self.create_grid(title=f"Epoch {epoch}", train=True)
                self.create_validation_grid(val_loader, title=f"Epoch {epoch}", train=True)
        
            if acc_g_loss/len(data_loader.dataset) < best_loss:
                best_loss = acc_g_loss/len(data_loader.dataset)
                torch.save(self.vae.state_dict(), os.path.join(models_dir, 'AdversarialVAE', f"AdvVAE_{self.dataset}.pt"))