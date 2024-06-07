####################################################################################################
### Code based on https://github.com/TeeyoHuang/conditional-GAN/blob/master/conditional_DCGAN.py ###
####################################################################################################

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
  if not os.path.exists(os.path.join(models_dir, 'ConditionalGAN')):
    os.makedirs(os.path.join(models_dir, 'ConditionalGAN'))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class Generator(nn.Module):
    # initializers
    def __init__(self, n_classes, latent_dim, d=128, channels=3):
        '''
        Generator model
        :param n_classes: number of classes
        :param latent_dim: latent dimension
        :param d: number of channels in the first layer
        :param channels: number of channels in the input image
        '''
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(latent_dim, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(n_classes, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, 4, 2, 1)
        self.latent_dim = latent_dim


    # forward method
    def forward(self, input, label):
        '''
        Forward pass
        :param input: input tensor
        :param label: label tensor
        :return: output tensor
        '''
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x
    
    @torch.no_grad()
    def sample(self, n_samples, device, n_classes):
        '''
        Sample from the generator
        :param n_samples: number of samples to generate
        :param device: device to run the model on
        :param n_classes: number of classes
        '''
        z = torch.randn(n_samples, self.latent_dim, 1, 1).to(device)
        labels = torch.randint(0, n_classes, (n_samples,))
        labels = F.one_hot(labels, n_classes).float().to(device).view(n_samples, n_classes, 1, 1)
        imgs = self.forward(z, labels)
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
    def __init__(self, n_classes, d=128, channels=3):
        '''
        Discriminator model
        :param n_classes: number of classes
        :param d: number of channels in the first layer
        :param channels: number of channels in the input image
        '''
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(n_classes, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)


    # def forward(self, input):
    def forward(self, input, label):
        '''
        Forward pass
        :param input: input tensor
        :param label: label tensor
        :return: output tensor
        '''
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x

class ConditionalGAN(nn.Module):
    def __init__(self, n_epochs, device, latent_dim, d, channels, lr, beta1, beta2, img_size, sample_and_save_freq, n_classes, dataset = 'mnist'):
        '''
        Conditional GAN model
        :param n_epochs: number of epochs to train the model
        :param device: device to run the model on
        :param latent_dim: latent dimension
        :param d: number of channels in the first layer
        :param channels: number of channels in the input image
        :param lr: learning rate
        :param beta1: beta1 parameter for Adam optimizer
        :param beta2: beta2 parameter for Adam optimizer
        :param img_size: size of the input image
        :param sample_and_save_freq: frequency to sample and save the images
        :param n_classes: number of classes
        :param dataset: dataset to train the model on
        '''
        super(ConditionalGAN, self).__init__()
        self.n_epochs = n_epochs
        self.device = device
        self.latent_dim = latent_dim
        self.d = d
        self.n_classes = n_classes
        self.channels = channels
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.img_size = img_size
        self.sample_and_save_freq = sample_and_save_freq
        self.generator = Generator(n_classes=self.n_classes, d=self.d, latent_dim = latent_dim, channels=channels).to(device)
        self.discriminator = Discriminator(n_classes=self.n_classes, d = self.d, channels=channels).to(device)
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.dataset = dataset

    def train_model(self, dataloader):
        '''
        Train the Conditional GAN model
        :param dataloader: data loader
        '''
        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        best_loss = np.inf

        epoch_bar = trange(self.n_epochs, desc = "Epochs", leave = True)

        create_checkpoint_dir()

        for epoch in epoch_bar:

            acc_g_loss = 0.0
            acc_d_loss = 0.0

            for (imgs, labels) in tqdm(dataloader, leave=False):

                # Adversarial ground truths
                valid = torch.ones(imgs.size(0), 1).to(self.device)
                fake = torch.zeros(imgs.size(0), 1).to(self.device)

                # Configure input
                real_imgs = imgs.to(self.device)
                # crete one hot vector with labels
                labels = F.one_hot(labels, self.n_classes).float().to(self.device).view(imgs.size(0), self.n_classes, 1, 1)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = torch.randn(imgs.size(0), self.latent_dim, 1, 1).to(self.device)
                gen_labels = torch.randint(0, self.n_classes, (imgs.size(0),))
                gen_labels = F.one_hot(gen_labels, self.n_classes).float().to(self.device).view(imgs.size(0), self.n_classes, 1, 1)

                # Generate a batch of images
                gen_imgs = self.generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                gen_labels_d = gen_labels.contiguous().expand(-1, -1, imgs.shape[2], imgs.shape[2])
                validity = self.discriminator(gen_imgs, gen_labels_d).view(-1, 1)
                g_loss = adversarial_loss(validity, valid)
                acc_g_loss += g_loss.item()*imgs.size(0)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                labels_d = labels.contiguous().expand(-1, -1, imgs.shape[2], imgs.shape[2])
                validity_real = self.discriminator(real_imgs, labels_d).view(-1, 1)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = self.discriminator(gen_imgs.detach(), gen_labels_d).view(-1, 1)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                acc_d_loss += d_loss.item()*imgs.size(0)

                d_loss.backward()
                optimizer_D.step()

            wandb.log({"Generator Loss": acc_g_loss/len(dataloader.dataset), "Discriminator Loss": acc_d_loss/len(dataloader.dataset)})
            epoch_bar.set_description("Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(acc_g_loss/len(dataloader.dataset), acc_d_loss/len(dataloader.dataset)))

            if acc_g_loss/len(dataloader.dataset) < best_loss:
                torch.save(self.generator.state_dict(), os.path.join(models_dir, 'ConditionalGAN', f"CondGAN_{self.dataset}.pt"))
                best_loss = acc_g_loss/len(dataloader.dataset)


            if epoch % self.sample_and_save_freq == 0:

                # create row of n_classes images
                z = torch.randn(self.n_classes, self.latent_dim, 1, 1).to(self.device)
                labels = torch.arange(self.n_classes).to(self.device)
                labels = F.one_hot(labels, self.n_classes).float().to(self.device).view(self.n_classes, self.n_classes, 1, 1)
                gen_imgs = self.generator(z, labels)
                gen_imgs = (gen_imgs + 1) / 2
                gen_imgs.clamp(0, 1)
                gen_imgs = gen_imgs.detach().cpu()

                # plot images
                fig = plt.figure(figsize=((self.n_classes//2 * 5), 5))
                grid = torchvision.utils.make_grid(gen_imgs, nrow=self.n_classes, normalize=True)
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis("off")
                wandb.log({"images": fig})
                plt.close(fig)