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
from sklearn.metrics import roc_auc_score, roc_curve

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'AdversarialVAE')):
    os.makedirs(os.path.join(models_dir, 'AdversarialVAE'))

class VanillaVAE(nn.Module):
    def __init__(self, input_shape, input_channels, latent_dim, hidden_dims = None, lr = 5e-3, batch_size = 64, kld_weight = 1e-4, loss_type = 'mse'):
        '''
        Vanilla VAE model
        Args:
        input_shape: Tuple with the input shape of the images
        input_channels: Number of channels of the input images
        latent_dim: Dimension of the latent space
        hidden_dims: List with the number of channels of the hidden layers of the VAE
        lr: Learning rate for the optimizer
        batch_size: Batch size for the training
        kld_weight: Weight for the KLD loss
        loss_type: Type of loss to use for the VAE. It can be 'mse' or 'bce'
        '''
        super(VanillaVAE, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.final_channels = input_channels
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.kld_weight = kld_weight
        self.mssim_loss = MSSIM(self.input_channels,
                                7,
                                True)
        self.loss_type = loss_type
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
        '''
        Encodes the input images
        Args:
        x: Input images
        '''
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)

        # Split into mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        '''
        Decodes the latent space
        Args:
        z: torch.Tensor, tensor with the latent space
        '''
        z = self.decoder_input(z)
        z = z.view(-1, self.last_channel, self.multiplier, self.multiplier)
        z = self.decoder(z)
        return self.final_layer(z)

    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        Args:
        mu: Mean of the latent space
        logvar: Log variance of the latent space
        Returns:
        z: Sample from the latent space
        '''
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        '''
        Forward pass of the model
        Args:
        x: Input images
        '''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, z):
        '''
        Generate samples from the latent space
        Args:
        z: torch.Tensor, tensor with the latent space
        '''
        return self.decode(z)
    
    def loss_function(self, recon_x, x, mu, logvar):
        '''Loss function of the model using MSE
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
        return mse + kld*self.kld_weight
    
    def ssim_loss_function(self, recon_x, x, mu, logvar):
        '''Loss function of the model using SSIM
        Args:
        recon_x: torch.Tensor, reconstructed input tensor
        x: torch.Tensor, input tensor
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        Returns:
        loss: torch.Tensor, loss of the model
        '''
        ssim = self.mssim_loss(recon_x*0.5 + 0.5,x*0.5 + 0.5)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1), dim=0)
        return ssim + kld*self.kld_weight
    

class Discriminator(nn.Module):
    def __init__(self, input_shape, input_channels, hidden_dims = None, lr = 5e-3, batch_size = 64):
        '''
        Discriminator model
        Args:
        input_shape: Tuple with the input shape of the images
        input_channels: Number of channels of the input images
        hidden_dims: List with the number of channels of the hidden layers of the discriminator
        lr: Learning rate for the optimizer
        batch_size: Batch size for the training
        '''
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
        '''
        Forward pass of the model
        Args:
        x: Input images
        '''
        x = self.encoder(x)
        return x

    def loss_function(self, x, y):
        '''
        Loss function of the model
        Args:
        x: Input images
        y: Labels
        Returns:
        loss: Loss of the model
        '''
        loss = nn.BCEWithLogitsLoss()
        return loss(x, y)
    
class AdversarialVAE(nn.Module):
    def __init__(self, input_shape, device, input_channels, latent_dim, n_epochs, hidden_dims = None, lr = 5e-3, batch_size = 64, gen_weight = 0.0005, recon_weight = 0.001, sample_and_save_frequency = 10, dataset = 'mnist', loss_type = 'mse', kld_weight = 1e-4):
        '''
        Adversarial VAE model
        Args:
        input_shape: Tuple with the input shape of the images
        device: Device to use for the model
        input_channels: Number of channels of the input images
        latent_dim: Dimension of the latent space
        n_epochs: Number of epochs to train the model
        hidden_dims: List with the number of channels of the hidden layers of the VAE
        lr: Learning rate for the optimizers
        batch_size: Batch size for the training
        gen_weight: Weight for the generator loss
        recon_weight: Weight for the reconstruction loss
        sample_and_save_frequency: Frequency to sample and save the images
        dataset: Name of the dataset
        loss_type: Type of loss to use for the VAE. It can be 'mse' or 'bce'
        kld_weight: Weight for the KLD loss
        '''
        super(AdversarialVAE, self).__init__()
        self.device = device
        self.vae = VanillaVAE(input_shape, input_channels, latent_dim, hidden_dims, lr, batch_size, kld_weight, loss_type).to(self.device)
        self.discriminator = Discriminator(input_shape, input_channels, hidden_dims, lr, batch_size).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.gen_weight = gen_weight
        self.recon_weight = recon_weight
        self.n_epochs = n_epochs
        self.sample_and_save_frequency = sample_and_save_frequency
        self.dataset = dataset
        self.loss_type = loss_type
        self.kld_weight = kld_weight

    def forward(self, x):
        '''
        Forward pass of the model
        Args:
        x: Input images
        Returns:
        image: Reconstructed images
        label: Discriminator output
        '''
        image,_,_ = self.vae(x)
        label = self.discriminator(image)
        return image, label
    
    def create_grid(self, figsize=(10, 10), title=None, train = False):
        '''
        Function to create a grid of samples
        Args:
        figsize: Tuple with the size of the figure
        title: Title of the figure
        train: If True, it will log the figure to wandb
        '''
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
        '''
        Function to create a grid of samples
        Args:
        data_loader: DataLoader with the validation data
        figsize: Tuple with the size of the figure
        title: Title of the figure
        train: If True, it will log the figure to wandb
        '''
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
        '''
        Function to train the model
        Args:
        data_loader: DataLoader with the training data
        val_loader: DataLoader with the validation data
        '''

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
                if self.loss_type == 'mse':
                    g_loss = self.recon_weight*adversarial_loss(validity_recon, valid) + self.gen_weight*adversarial_loss(validity_gen, valid) + self.vae.loss_function(recon_imgs, real_imgs, mu, logvar)
                else:
                    g_loss = self.recon_weight*adversarial_loss(validity_recon, valid) + self.gen_weight*adversarial_loss(validity_gen, valid) + self.vae.ssim_loss_function(recon_imgs, real_imgs, mu, logvar)
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

                # Loss for reconstructed images
                validity_recon = self.discriminator(recon_imgs.detach())
                d_recon_loss = adversarial_loss(validity_recon, fake)

                # Total self.discriminator loss
                d_loss = (d_real_loss + d_fake_loss*0.5 + d_recon_loss*0.5) / 2
                acc_d_loss += d_loss.item()*imgs.size(0)

                d_loss.backward()
                optimizer_D.step()

            epochs_bar.set_description(f"Loss: {acc_g_loss/len(data_loader.dataset):.4f} - D Loss: {acc_d_loss/len(data_loader.dataset):.4f}")
            wandb.log({"Generator Loss": acc_g_loss/len(data_loader.dataset), "Discriminator Loss": acc_d_loss/len(data_loader.dataset)})


            if (epoch+1) % self.sample_and_save_frequency == 0 or epoch == 0:
                self.create_grid(title=f"Epoch {epoch}", train=True)
                self.create_validation_grid(val_loader, title=f"Epoch {epoch}", train=True)
                torch.save(self.discriminator.state_dict(), os.path.join(models_dir, 'AdversarialVAE', f"Discriminator_{self.dataset}_{epoch}.pt"))
        
            if acc_g_loss/len(data_loader.dataset) < best_loss:
                best_loss = acc_g_loss/len(data_loader.dataset)
                torch.save(self.vae.state_dict(), os.path.join(models_dir, 'AdversarialVAE', f"AdvVAE_{self.dataset}.pt"))
                torch.save(self.discriminator.state_dict(), os.path.join(models_dir, 'AdversarialVAE', f"Discriminator_{self.dataset}.pt"))
        '''
        # create an artifact for the state dict
        artifact = wandb.Artifact(f"AdvVAE_{self.dataset}", type="model")
        artifact.add_file(os.path.join(models_dir, 'AdversarialVAE', f"AdvVAE_{self.dataset}.pt"))
        # save the artifact to wandb
        wandb.log_artifact(artifact)
        # create an artifact for the state dict
        artifact = wandb.Artifact(f"Discriminator_{self.dataset}", type="model")
        artifact.add_file(os.path.join(models_dir, 'AdversarialVAE', f"Discriminator_{self.dataset}.pt"))
        # save the artifact to wandb
        wandb.log_artifact(artifact)
        '''

    def ood_score(self, recon_x, x, mu, logvar):
        '''
        Function to compute the OOD score
        Args:
        recon_x: Reconstructed images
        x: Input images
        mu: Mean of the latent space
        logvar: Log variance of the latent space
        Returns:
        mse: Mean squared error
        kld: Kullback-Leibler divergence
        discriminator_score: Score of the discriminator
        '''
        if self.loss_type == 'mse':
            loss_mse = nn.MSELoss(reduction='none')
            loss_adv = nn.BCELoss(reduction='none')
            mse = loss_mse(x, recon_x).mean(dim=(1,2,3))
            kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1)
            discriminator_score = loss_adv(self.discriminator(recon_x), torch.ones(x.size(0), 1).to(self.device)).squeeze()
            return mse + kld*self.kld_weight + self.recon_weight*discriminator_score
        else:
            # iterate over the batch
            ssim = torch.zeros(recon_x.size(0), device = self.device)
            for i in range(recon_x.size(0)):
                ssim[i] = self.mssim_loss(recon_x[i].unsqueeze(0)*0.5 + 0.5,x[i].unsqueeze(0)*0.5 + 0.5)
            loss_adv = nn.BCELoss(reduction='none')
            kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1)
            discriminator_score = loss_adv(self.discriminator(recon_x), torch.ones(x.size(0), 1).to(self.device)).squeeze()
            return ssim + kld*self.kld_weight + self.recon_weight*discriminator_score


    def outlier_detection(self, in_loader, out_loader, display = True, in_array = None, in_array_discriminator = None):
        '''
        Function to test the outlier detection capabilities of the model
        Args:
        in_loader: DataLoader with the in-distribution data
        out_loader: DataLoader with the out-of-distribution data
        display: If True, it will display the histograms of the scores
        in_array: If not None, it will use this array of in-distribution scores instead of computing them
        in_array_discriminator: If not None, it will use this array of in-distribution discriminator scores instead of computing them
        Returns:
        in_scores: Array with the in-distribution scores
        in_scores_discriminator: Array with the in-distribution discriminator scores
        rocauc: ROC AUC score of the model
        rocauc_discriminator: ROC AUC score of the discriminator
        '''
        in_scores = []
        in_scores_discriminator = []
        out_scores = []
        out_scores_discriminator = []
        if in_array is not None and in_array_discriminator is not None:
            in_scores = in_array
            in_scores_discriminator = in_array_discriminator
        else:
            for (imgs, _) in tqdm(in_loader, desc = 'In-distribution', leave=False):
                in_scores_discriminator.append(self.discriminator(imgs.to(self.device)).detach().cpu().numpy())
                recon_imgs, mu, logvar = self.vae(imgs.to(self.device))
                in_scores.append(self.ood_score(recon_imgs, imgs.to(self.device), mu, logvar).detach().cpu().numpy())

            in_scores = np.concatenate(in_scores)
            in_scores_discriminator = np.concatenate(in_scores_discriminator)
            in_scores_discriminator = -in_scores_discriminator + 1

        for (imgs, _) in tqdm(out_loader, desc = 'Out-of-distribution', leave=False):
            out_scores_discriminator.append(self.discriminator(imgs.to(self.device)).detach().cpu().numpy())
            recon_imgs, mu, logvar = self.vae(imgs.to(self.device))
            out_scores.append(self.ood_score(recon_imgs, imgs.to(self.device), mu, logvar).detach().cpu().numpy())

        out_scores = np.concatenate(out_scores)
        out_scores_discriminator = np.concatenate(out_scores_discriminator)
        out_scores_discriminator = -out_scores_discriminator + 1

        rocauc = roc_auc_score(np.concatenate([np.zeros_like(in_scores), np.ones_like(out_scores)]), np.concatenate([in_scores, out_scores]))
        rocauc_discriminator = roc_auc_score(np.concatenate([np.zeros_like(in_scores_discriminator), np.ones_like(out_scores_discriminator)]), np.concatenate([in_scores_discriminator, out_scores_discriminator]))

        fpr, tpr , _ = roc_curve(np.concatenate([np.zeros_like(in_scores), np.ones_like(out_scores)]), np.concatenate([in_scores, out_scores]))
        fpr_discriminator, tpr_discriminator , _ = roc_curve(np.concatenate([np.zeros_like(in_scores_discriminator), np.ones_like(out_scores_discriminator)]), np.concatenate([in_scores_discriminator, out_scores_discriminator]))

        fpr95 = fpr[np.argmax(tpr >= 0.95)]
        fpr95_discriminator = fpr_discriminator[np.argmax(tpr_discriminator >= 0.95)]

        if display:
            # print discriminator metrics
            print(f"ROC AUC Discriminator: {rocauc_discriminator:.6f}, FPR95 Discriminator: {fpr95_discriminator:.6f}, Mean Scores Discriminator: {np.mean(out_scores_discriminator):.6f}, ROC AUC VAE: {rocauc:.6f}, FPR95 VAE: {fpr95:.6f}")
            print(f"Mean Scores ID: {np.mean(in_scores_discriminator):.6f}")
            # plot the scores
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].hist(in_scores, bins=100, alpha=0.5, label='In-distribution')
            ax[0].hist(out_scores, bins=100, alpha=0.5, label='Out-of-distribution')
            ax[0].set_title('VAE Scores (ROC AUC: {:.4f})'.format(rocauc))
            ax[0].legend()
            ax[1].hist(in_scores_discriminator, bins=50, alpha=0.5, label='In-distribution')
            ax[1].hist(out_scores_discriminator, bins=50, alpha=0.5, label='Out-of-distribution')
            ax[1].set_title('Discriminator Scores (ROC AUC: {:.4f})'.format(rocauc_discriminator))
            ax[1].legend()
            plt.show()
        
        else:
            return in_scores, in_scores_discriminator, rocauc, rocauc_discriminator, fpr95_discriminator, np.mean(out_scores_discriminator)
        

class MSSIM(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 window_size: int=11,
                 size_average:bool = True) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)

        Args:
        in_channels: int, number of channels of the input image
        window_size: int, size of the window for the SSIM
        size_average: bool, if the loss should be averaged
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size:int, sigma: float):
        """
        Generates a gaussian window
        Args:
        window_size: int, size of the window
        sigma: float, standard deviation of the gaussian
        Returns:
        kernel: torch.Tensor, gaussian window
        """
        kernel = torch.tensor([exp((x - window_size // 2)**2/(2 * sigma ** 2))
                               for x in range(window_size)])
        return kernel/kernel.sum()

    def create_window(self, window_size, in_channels):
        """
        Creates a 2D window for the SSIM
        Args:
        window_size: int, size of the window
        in_channels: int, number of channels of the input image
        Returns:
        window: torch.Tensor, 2D window
        """
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self,
             img1,
             img2,
             window_size: int,
             in_channel: int,
             size_average: bool):
        """
        Computes the SSIM
        Args:
        img1: torch.Tensor, input image
        img2: torch.Tensor, input image
        window_size: int, size of the window
        in_channel: int, number of channels of the input image
        size_average: bool, if the loss should be averaged
        Returns:
        ret: torch.Tensor, SSIM loss
        """

        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = F.conv2d(img1, window, padding= window_size//2, groups=in_channel)
        mu2 = F.conv2d(img2, window, padding= window_size//2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding = window_size//2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding = window_size//2, groups=in_channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding = window_size//2, groups=in_channel) - mu1_mu2

        img_range = 1.0 #img1.max() - img1.min() # Dynamic range
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, img1, img2):
        """
        Computes the MS-SSIM
        Args:
        img1: torch.Tensor, input image
        img2: torch.Tensor, input image
        Returns:
        output: torch.Tensor, MS-SSIM loss
        """
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        # if normalize:
        #     mssim = (mssim + 1) / 2
        #     mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output
