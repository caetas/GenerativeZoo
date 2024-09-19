######################################################################################
### Code based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py ###
######################################################################################

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
  if not os.path.exists(os.path.join(models_dir, 'ConditionalVAE')):
    os.makedirs(os.path.join(models_dir, 'ConditionalVAE'))

class ConditionalVAE(nn.Module):
    def __init__(self, input_shape, input_channels, args):
        '''Conditional VAE model
        Args:
        input_shape: int, input shape of the image
        input_channels: int, number of channels of the input image
        args: argparse.ArgumentParser, arguments for the model
        '''
        super(ConditionalVAE, self).__init__()
        self.no_wandb = args.no_wandb
        self.num_classes = args.num_classes
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.final_channels = input_channels
        self.latent_dim = args.latent_dim
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_and_save_freq = args.sample_and_save_freq
        self.dataset = args.dataset
        self.mssim_loss = MSSIM(self.input_channels,
                                7,
                                True)
        self.kld_weight = args.kld_weight
        self.loss_type = args.loss_type

        self.embed_class = nn.Linear(self.num_classes, self.input_shape**2)
        self.embed_data = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=1)
        self.hidden_dims = args.hidden_dims

        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims_str = '_'.join(map(str, self.hidden_dims))
        
        # each layer decreases the h and w by 2, so we need to divide by 2**(number of layers) to know the factor for the flattened input
        self.multiplier = int(self.input_shape/(2**len(self.hidden_dims)))
        self.last_channel = self.hidden_dims[-1]
        modules = []

        # to account for classes
        input_channels += 1

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, h_dim, kernel_size = 3, stride = 2, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            input_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*(self.multiplier**2), self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1]*(self.multiplier**2), self.latent_dim)

        modules = []

        self.decoder_input = nn.Linear(self.latent_dim + self.num_classes, self.hidden_dims[-1]*(self.multiplier**2))

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1], kernel_size=3, stride = 2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=self.final_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.encoder.to(self.device)
        self.fc_mu.to(self.device)
        self.fc_logvar.to(self.device)
        self.decoder_input.to(self.device)
        self.decoder.to(self.device)
        self.final_layer.to(self.device)
        self.embed_class.to(self.device)
        self.embed_data.to(self.device)

    def encode(self, x):
        '''Encode the input data
        Args:
        x: torch.Tensor, input data
        Returns:
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        '''
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)
        return self.fc_mu(x), self.fc_logvar(x)
    
    def decode(self, z):
        '''Decode the latent space
        Args:
        z: torch.Tensor, latent space
        Returns:
        torch.Tensor, reconstructed data
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
        torch.Tensor, reparameterized latent space
        '''
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + (eps*std)
    
    def forward(self, x, y):
        '''Forward pass
        Args:
        x: torch.Tensor, input data
        y: torch.Tensor, conditional data
        Returns:
        torch.Tensor, reconstructed data
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        '''
        x = self.embed_data(x)
        y_emb = self.embed_class(y)
        y_emb = y_emb.view(-1,self.input_shape, self.input_shape).unsqueeze(1)
        x = torch.cat([x, y_emb], dim = 1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, y], dim = 1)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        '''Loss function for the model
        Args:
        recon_x: torch.Tensor, reconstructed data
        x: torch.Tensor, input data
        mu: torch.Tensor, mean of the latent space
        logvar: torch.Tensor, logvar of the latent space
        Returns:
        torch.Tensor, loss of the model
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
    
    def generate(self, z, y):
        '''Generate samples from the latent space
        Args:
        z: torch.Tensor, latent space
        y: torch.Tensor, conditional data
        Returns:
        torch.Tensor, reconstructed data
        '''
        z = torch.cat([z, y], dim = 1)
        return self.decode(z)
    
    def one_hot_encode(self, y):
        '''One hot encode the conditional data
        Args:
        y: torch.Tensor, conditional data
        Returns:
        torch.Tensor, one hot encoded conditional data
        '''
        y_onehot = torch.zeros(y.size(0), self.num_classes)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)
        return y_onehot
    
    def sample(self, figsize=(10, 10), title=None, train = False, n = 16, label = None):
        '''Create a grid of samples from the latent space
        Args:
        device: torch.device to run the model
        figsize: tuple, size of the figure
        title: str, title of the figure
        n: int, number of samples
        label: int, label of the class to generate samples from
        Returns:
        torch.Tensor, grid of samples
        '''
        z = torch.randn(n, self.latent_dim).to(self.device)
        if label is None:
            y = torch.eye(n=n, m=self.num_classes).to(self.device)
        else:
            y = label*torch.ones(n).to(self.device)
            y = F.one_hot(y, num_classes=self.num_classes).float()
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
        if train:
            if not self.no_wandb:
                wandb.log({'Samples': fig})
        else:
            plt.show()
        plt.close(fig)
        return grid
    
    def train_model(self, train_loader, epochs, verbose = True):
        '''Train the model
        Args:
        train_loader: torch.utils.data.DataLoader, dataloader for the training
        epochs: int, number of epochs for the training
        device: torch.device to run the model
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        self.train()
        epochs_bar = trange(epochs)

        create_checkpoint_dir()

        best_loss = np.inf

        for epoch in epochs_bar:
            acc_loss = 0
            for x, y in tqdm(train_loader, leave=False, desc='Batches', disable=not verbose):
                x = x.to(self.device)
                y = self.one_hot_encode(y).float().to(self.device)
                recon_x, mu, logvar = self(x, y)
                if self.loss_type == 'mse':
                    loss = self.loss_function(recon_x, x, mu, logvar)
                else:
                    loss = self.ssim_loss_function(recon_x, x, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc_loss += loss.item()
            epochs_bar.set_description(f"Loss: {acc_loss/len(train_loader.dataset):.8f}")
            epochs_bar.refresh()
            if not self.no_wandb:
                wandb.log({"loss": acc_loss/len(train_loader.dataset)})
            if epoch % self.sample_and_save_freq == 0:
                self.sample(title=f"Epoch_{epoch}", train = True)
            
            if acc_loss<best_loss:
                best_loss = acc_loss
                torch.save(self.state_dict(), os.path.join(models_dir,'ConditionalVAE', f"CondVAE_{self.dataset}_{self.latent_dim}_{self.hidden_dims_str}_{self.loss_type}.pt"))


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