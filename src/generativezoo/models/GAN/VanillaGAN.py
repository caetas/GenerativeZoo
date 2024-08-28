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
from sklearn.metrics import roc_auc_score, roc_curve

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
    def __init__(self, latent_dim, d=128, channels=3, imgSize=32):
        '''
        Generator model
        :param latent_dim: latent dimension
        :param d: number of channels in the first layer
        :param channels: number of channels in the output image
        :param imgSize: size of the output image
        '''
        super(Generator, self).__init__()
        if imgSize < 64:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     latent_dim, d * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.ReLU(True),
                # state size. (d*4) x 4 x 4
                nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.ReLU(True),
                # state size. (d*2) x 8 x 8
                nn.ConvTranspose2d(d * 2, d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(True),
                # state size. (d) x 16 x 16
                nn.ConvTranspose2d(    d,      channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (channels) x 32 x 32
            )
        elif imgSize == 64:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     latent_dim, d * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(d * 8),
                nn.ReLU(True),
                # state size. (d*8) x 4 x 4
                nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.ReLU(True),
                # state size. (d*4) x 8 x 8
                nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.ReLU(True),
                # state size. (d*2) x 16 x 16
                nn.ConvTranspose2d(d * 2,    d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(True),
                # state size. (d) x 32 x 32
                nn.ConvTranspose2d(    d,      channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (channels) x 64 x 64
            )
        
        elif imgSize > 64:
            # take input of size batch_size x latent_dim, 1, 1 and reshape it to batch_size x d*8*imgSize//64*imgSize//64 x 1 x 1
            self.reshape = nn.Linear(latent_dim, d*16*imgSize//32*imgSize//32)
            self.main = nn.Sequential(
                # state size. (d*16) x 4 x 4
                nn.ConvTranspose2d(d*16, d * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 8),
                nn.ReLU(True),
                # state size. (d*8) x 8 x 8
                nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.ReLU(True),
                # state size. (d*4) x 16 x 16
                nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.ReLU(True),
                # state size. (d*2) x 32 x 32
                nn.ConvTranspose2d(d * 2,    d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(True),
                # state size. (d) x 64 x 64
                nn.ConvTranspose2d(    d,      channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (channels) x 128 x 128
            )

        self.imgSize = imgSize
        self.latent_dim = latent_dim

    # forward method
    def forward(self, input):
        '''
        Forward pass of the generator
        :param input: input tensor
        '''
        if self.imgSize > 64:
            input = input.view(input.size(0), -1)
            input = self.reshape(input)
            input = input.view(input.size(0), -1, self.imgSize//32, self.imgSize//32)
        x = self.main(input)
        return x
    
    @torch.no_grad()
    def sample(self, n_samples, device):
        '''
        Generate samples from the model
        :param n_samples: number of samples to generate
        :param device: device to use
        '''
        self.eval()
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
    def __init__(self, d=128, channels=3, imgSize=32):
        '''
        Discriminator model
        :param d: number of channels in the first layer
        :param channels: number of channels in the input image
        :param imgSize: size of the input image
        '''
        super(Discriminator, self).__init__()
        if imgSize < 64:
            self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(channels, d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d) x 16 x 16
                nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*2) x 8 x 8
                nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*4) x 4 x 4
                nn.Conv2d(d * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        elif imgSize == 64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(channels, d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d) x 32 x 32
                nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*2) x 16 x 16
                nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*4) x 8 x 8
                nn.Conv2d(d * 4, d * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*8) x 4 x 4
                nn.Conv2d(d * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        
        elif imgSize >= 64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(channels, d, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d) x 32 x 32
                nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*2) x 16 x 16
                nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*4) x 8 x 8
                nn.Conv2d(d * 4, d * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (d*8) x 4 x 4
                nn.Conv2d(d * 8, 1, 4, 1, 0, bias=False),
                nn.Flatten(),
                nn.Linear((imgSize//16 - 3)**2, 1),
                nn.Sigmoid()
            )


    # def forward(self, input):
    def forward(self, input):
        '''
        Forward pass of the discriminator
        :param input: input tensor
        '''
        x = self.main(input)
        return x
    
    @torch.no_grad()
    def outlier_detection(self, in_loader, out_loader, device, in_array=None, display=True):
        '''
        Detect outliers using the discriminator
        :param in_loader: dataloader for in-distribution data
        :param out_loader: dataloader for out-of-distribution data
        :param device: device to use
        :param in_array: in-distribution predictions
        :param display: whether to display the results
        '''
        self.eval()

        in_preds = []
        out_preds = []

        if in_array is None:
            for (imgs, _) in tqdm(in_loader, desc='In-distribution', leave=False):
                imgs = imgs.to(device)
                preds = self.forward(imgs)
                preds = preds.cpu().numpy()
                if len(preds.shape) > 2:
                    in_preds.append(preds[:,0,0,0])
                else:
                    in_preds.append(preds[:,0])
            in_array = np.concatenate(in_preds)
            in_array = -in_array + 1

        else:
            in_preds = in_array

        for (imgs,_) in tqdm(out_loader, desc='Out-of-distribution', leave=False):
            imgs = imgs.to(device)
            preds = self.forward(imgs)
            preds = preds.cpu().numpy()
            if len(preds.shape) > 2:
                out_preds.append(preds[:,0,0,0])
            else:
                out_preds.append(preds[:,0])

        out_array = np.concatenate(out_preds)

        out_array = -out_array + 1
        labels = np.concatenate([np.zeros(in_array.shape[0]), np.ones(out_array.shape[0])])


        # calculate auroc
        preds = np.concatenate([in_array, out_array])
        auroc = roc_auc_score(labels, preds)

        fpr, tpr, _ = roc_curve(labels, preds)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]

        if display:
            print(f"AUROC: {auroc:.8f}, FPR95: {fpr95:.8f}, Mean Scores: {np.mean(out_array):.8f}")
            plt.hist(in_array, bins=100, alpha=0.5, label='In-distribution')
            plt.hist(out_array, bins=100, alpha=0.5, label='Out-of-distribution')
            plt.legend()
            plt.show()

        return auroc, fpr95, in_array, np.mean(out_array)
    
class VanillaGAN(nn.Module):
    def __init__(self, args, channels=3, img_size = 32):
        '''
        Vanilla GAN model
        :param args: arguments
        :param channels: number of channels in the image
        :param img_size: size of the image
        '''
        super(VanillaGAN, self).__init__()
        self.n_epochs = args.n_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(latent_dim = args.latent_dim, channels=channels, imgSize=img_size, d=args.d).to(self.device)
        self.discriminator = Discriminator(channels=channels, d=args.d, imgSize=img_size).to(self.device)
        self.latent_dim = args.latent_dim
        self.d = args.d
        self.channels = channels
        self.lrg = args.lrg
        self.lrd = args.lrd
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.img_size = img_size
        self.sample_and_save_freq = args.sample_and_save_freq
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.dataset = args.dataset
        self.no_wandb = args.no_wandb
    
    def train_model(self, dataloader, verbose = True):
        '''
        Train the model
        :param dataloader: dataloader for the data
        '''
        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lrg, betas=(self.beta1, self.beta2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lrd, betas=(self.beta1, self.beta2))

        epochs_bar = trange(self.n_epochs, desc = "Loss: ----", leave = True)
        best_loss = np.inf

        create_checkpoint_dir()

        for epoch in epochs_bar:

            acc_g_loss = 0.0
            acc_d_loss = 0.0

            for (imgs, _) in tqdm(dataloader, leave=False, desc='Batches', disable=not verbose):

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

            if not self.no_wandb:
                wandb.log({"Generator Loss": acc_g_loss/len(dataloader.dataset), "Discriminator Loss": acc_d_loss/len(dataloader.dataset)})
            epochs_bar.set_description("Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(acc_g_loss/len(dataloader.dataset), acc_d_loss/len(dataloader.dataset)))
            
            if acc_g_loss/len(dataloader.dataset) < best_loss and epoch >= 20:
                best_loss = acc_g_loss/len(dataloader.dataset)
                torch.save(self.generator.state_dict(), os.path.join(models_dir, 'VanillaGAN', f"VanGAN_{self.dataset}.pt"))
                torch.save(self.discriminator.state_dict(), os.path.join(models_dir, 'VanillaGAN', f"VanDisc_{self.dataset}.pt"))

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
                if not self.no_wandb:
                    wandb.log({"Generated Images": fig})
                plt.close(fig)
                torch.save(self.discriminator.state_dict(), os.path.join(models_dir, 'VanillaGAN', f"VanDisc_{self.dataset}_{epoch}.pt"))
