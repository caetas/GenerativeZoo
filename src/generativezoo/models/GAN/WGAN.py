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
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import wandb
from sklearn.metrics import roc_auc_score, roc_curve

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'WassersteinGAN')):
    os.makedirs(os.path.join(models_dir, 'WassersteinGAN'))

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
                #nn.Tanh()
                nn.Sigmoid()
                # state size. (channels) x 32 x 32
            )
        
        elif imgSize >= 64:
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
                #nn.Tanh()
                nn.Sigmoid()
                # state size. (channels) x 128 x 128
            )

        self.imgSize = imgSize
        self.latent_dim = latent_dim

    # forward method
    def forward(self, input):
        if self.imgSize >= 64:
            input = input.view(input.size(0), -1)
            input = self.reshape(input)
            input = input.view(input.size(0), -1, self.imgSize//32, self.imgSize//32)
        x = self.main(input)
        return x
    
    @torch.no_grad()
    def sample(self, n_samples, device, train = True):
        z = torch.randn(n_samples, self.latent_dim, 1, 1).to(device)
        imgs = self.forward(z)
        #imgs = (imgs + 1) / 2
        imgs = imgs.detach().cpu()
        # create a grid of sqrt(n_samples) x sqrt(n_samples) images
        grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(n_samples)), normalize=True)
        fig = plt.figure(figsize=(10, 10))
        # make an image from the grid
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        if not train:
            plt.show()
        else:
            wandb.log({"Generated Images": fig})
        plt.close(fig)

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, channels=3, imgSize=32):
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
                nn.Flatten(),
                nn.Linear((imgSize//8 - 3)**2, 1),
                #nn.Sigmoid()
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
                #nn.Sigmoid()
            )

    # def forward(self, input):
    def forward(self, input):
        x = self.main(input)
        return x
    

def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, 'WassersteinGAN')):
        os.makedirs(os.path.join(models_dir, 'WassersteinGAN'))

class WGAN(nn.Module):
    def __init__(self, latent_dim, d=64, channels=3, imgSize=32, batch_size=64, n_epochs = 100, gp_weight=10, n_critic=5, dataset='cifar10', sample_and_save_freq = 5, lrd=0.0002, lrg=0.0002, beta1=0.5, beta2=0.999):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        self.G = Generator(latent_dim, d, channels, imgSize)
        self.D = Discriminator(d, channels, imgSize)
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lrg, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lrd, betas=(beta1, beta2))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device)
        self.D.to(self.device)
        self.batch_size = batch_size
        self.imgSize = imgSize
        self.channels = channels
        self.d = d
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = n_critic
        self.dataset = dataset
        self.sample_and_save_freq = sample_and_save_freq
        self.n_epochs = n_epochs
        self.lrd = lrd
        self.lrg = lrg
        self.beta1 = beta1
        self.beta2 = beta2

    def _gradient_penalty(self, real, fake):
        batch_size = real.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real).to(self.device)

        interpolated = alpha * real.data + (1 - alpha) * fake.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    def train_model(self, dataloader):
        epoch_bar = trange(self.n_epochs, desc='Epochs', leave=True)
        best_loss = np.inf

        create_checkpoint_dir()

        for epoch in epoch_bar:
            acc_loss = 0
            acc_loss_d = 0
            cnt = 0
            cnt_d = 0
            for img,_ in tqdm(dataloader, desc='Batches', leave=False):
                self.num_steps += 1
                real_imgs = img.to(self.device)
                batch_size = real_imgs.size()[0]
                # Sample noise
                z = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)

                # Generate images
                fake_imgs = self.G(z)

                # Train the discriminator
                self.optimizer_D.zero_grad()

                # Real images
                real_validity = self.D(real_imgs)
                # Fake images
                fake_validity = self.D(fake_imgs)

                # Gradient penalty
                gradient_penalty = self._gradient_penalty(real_imgs, fake_imgs)

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()

                #self.losses['D'].append(d_loss.data[0])
                #self.losses['GP'].append(gradient_penalty.data[0])
                acc_loss_d += d_loss.item()*batch_size
                cnt_d += batch_size

                # Train the generator every n_critic iterations
                if self.num_steps % self.critic_iterations == 0:
                    self.optimizer_G.zero_grad()

                    # Generate a batch of images
                    fake_imgs = self.G(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.D(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()

                    #self.losses['G'].append(g_loss.data[0])
                    acc_loss += g_loss.item()*batch_size
                    cnt =+ batch_size

            epoch_bar.set_postfix({'Generator Loss': acc_loss/cnt})
            wandb.log({"Generator Loss": acc_loss/cnt, "Discriminator Loss": acc_loss_d/cnt_d})
            if acc_loss/cnt < best_loss:
                torch.save(self.G.state_dict(), os.path.join(models_dir, 'WassersteinGAN', f'WGAN_{self.dataset}.pt'))
                torch.save(self.D.state_dict(), os.path.join(models_dir, 'WassersteinGAN', f'WGAN_{self.dataset}_D.pt'))
                best_loss = acc_loss/cnt

            if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:
                self.G.sample(16, self.device, train = True)

    @torch.no_grad()
    def outlier_detection(self, in_loader, out_loader, in_array = None, display=True):
        self.D.eval()
        out_preds = []
        if in_array is None:
            in_preds = []
            for img,_ in tqdm(in_loader, desc='Inlier Detection', leave=True):
                img = img.to(self.device)
                pred = self.D(img)
                if len(pred.size()) > 2:
                    in_preds.append(pred.cpu().numpy()[:,0,0,0])
                else:
                    in_preds.append(pred.cpu().numpy()[:,0])
            in_preds = np.concatenate(in_preds)
            in_preds = -in_preds + 1
        else:
            in_preds = in_array

        for img, _ in tqdm(out_loader, desc='Outlier Detection', leave=True):
            img = img.to(self.device)
            pred = self.D(img)
            if len(pred.size()) > 2:
                out_preds.append(pred.cpu().numpy()[:,0,0,0])
            else:
                out_preds.append(pred.cpu().numpy()[:,0])

        out_preds = np.concatenate(out_preds)
        out_preds = -out_preds + 1

        labels = np.concatenate([np.zeros_like(in_preds), np.ones_like(out_preds)])
        preds = np.concatenate([in_preds, out_preds])

        fpr, tpr, _ = roc_curve(labels, preds)
        auc = roc_auc_score(labels, preds)
        fpr95 = fpr[np.argmax(tpr >= 0.95)]

        if display:
            plt.figure(figsize=(10, 10))
            plt.hist(in_preds, bins=50, alpha=0.5, label='Inlier')
            plt.hist(out_preds, bins=50, alpha=0.5, label='Outlier')
            plt.legend()
            plt.show()

        return auc, fpr95, in_preds, out_preds