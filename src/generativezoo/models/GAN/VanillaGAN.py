###################################################################################################
### Code based onhttps://github.com/TeeyoHuang/conditional-GAN/blob/master/conditional_DCGAN.py ###
###################################################################################################

from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import wandb
import torchvision
import numpy as np
from config import models_dir
import os

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'VanillaGAN')):
    os.makedirs(os.path.join(models_dir, 'VanillaGAN'))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dim, d=128, channels=3):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, d*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, 4, 2, 1)
        self.latent_dim = latent_dim


    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x
    
    @torch.no_grad()
    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.latent_dim, 1, 1).to(device)
        imgs = self.forward(z)
        imgs = (imgs + 1) / 2
        imgs = imgs.detach().cpu()
        # create a grid of sqrt(n_samples) x sqrt(n_samples) images
        grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(n_samples)), normalize=True)
        # make an image from the grid
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)


    # def forward(self, input):
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x
    
class VanillaGAN(nn.Module):
    def __init__(self, n_epochs, device, latent_dim, d=128, channels=3, lr = 0.0002, beta1 = 0.5, beta2 = 0.999, img_size = 32, sample_and_save_freq = 5, dataset = 'mnist'):
        super(VanillaGAN, self).__init__()
        self.n_epochs = n_epochs
        self.device = device
        self.generator = Generator(latent_dim = latent_dim, channels=channels).to(self.device)
        self.discriminator = Discriminator(channels=channels, d=d).to(self.device)
        self.latent_dim = latent_dim
        self.d = d
        self.channels = channels
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.img_size = img_size
        self.sample_and_save_freq = sample_and_save_freq
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.dataset = dataset
    
    def train_model(self, dataloader):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        epochs_bar = trange(self.n_epochs, desc = "Loss: ----", leave = True)
        best_loss = np.inf

        create_checkpoint_dir()

        for epoch in epochs_bar:

            acc_g_loss = 0.0
            acc_d_loss = 0.0

            for (imgs, _) in tqdm(dataloader, leave=False):

                # Adversarial ground truths
                valid = torch.ones(imgs.size(0), 1).to(self.device)
                fake = torch.zeros(imgs.size(0), 1).to(self.device)

                # Configure input
                real_imgs = imgs.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = torch.randn(imgs.size(0), self.latent_dim, 1, 1).to(self.device)

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                validity = self.discriminator(gen_imgs).view(-1, 1)
                g_loss = adversarial_loss(validity, valid)
                acc_g_loss += g_loss.item()*imgs.size(0)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = self.discriminator(real_imgs).view(-1, 1)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = self.discriminator(gen_imgs.detach()).view(-1, 1)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                acc_d_loss += d_loss.item()*imgs.size(0)

                d_loss.backward()
                optimizer_D.step()
            
            wandb.log({"Generator Loss": acc_g_loss/len(dataloader.dataset), "Discriminator Loss": acc_d_loss/len(dataloader.dataset)})
            epochs_bar.set_description("Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(acc_g_loss/len(dataloader.dataset), acc_d_loss/len(dataloader.dataset)))
            
            if acc_g_loss/len(dataloader.dataset) < best_loss:
                best_loss = acc_g_loss/len(dataloader.dataset)
                torch.save(self.generator.state_dict(), os.path.join(models_dir, 'VanillaGAN', f"VanGAN_{self.dataset}.pt"))

            if epoch % self.sample_and_save_freq == 0:
                # create row of n_classes images
                z = torch.randn(16, self.latent_dim, 1, 1).to(self.device)
                gen_imgs = self.generator(z)
                gen_imgs = (gen_imgs + 1) / 2
                gen_imgs.clamp(0, 1)
                # plot images
                fig = plt.figure(figsize=(10, 10))
                # create a grid of sqrt(n_samples) x sqrt(n_samples) images
                grid = torchvision.utils.make_grid(gen_imgs.detach().cpu(), nrow=4, normalize=True)
                # make an image from the grid
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis('off')
                wandb.log({"Generated Images": fig})
                plt.close(fig)