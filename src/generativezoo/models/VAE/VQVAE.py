import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import os
from config import figures_dir, models_dir

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, encoding_inds  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input):
        return input + self.resblock(input)

class VQVAE(nn.Module):
    def __init__(self, input_shape, input_channels, embedding_dim, num_embeddings, hidden_dims = None, lr = 1e-3, batch_size = 64, beta=0.25):
        super(VQVAE, self).__init__()

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.final_channels = input_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size

        if hidden_dims is None:
            hidden_dims = [128, 256]
        
        # each layer decreases the h and w by 2, so we need to divide by 2**(number of layers) to know the factor for the flattened input
        self.multiplier = int(2**len(hidden_dims))
        self.last_channel = hidden_dims[-1]
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, h_dim, kernel_size = 4, stride = 2, padding = 1),
                    nn.LeakyReLU()
                )
            )
            input_channels = h_dim
        
        modules.append(
            nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(
                ResidualLayer(input_channels, input_channels)
            )
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(input_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(
                ResidualLayer(hidden_dims[-1], hidden_dims[-1])
            )
        
        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride = 2, padding=1),
                    nn.LeakyReLU())
            )
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], self.final_channels, kernel_size=4, stride = 2, padding=1),
                nn.Tanh())
        )

        self.decoder = nn.Sequential(*modules)

    
    def encode(self, x):
        return self.encoder(x)
    
    def quantize(self, z):
        return self.vq_layer(z)
    
    def decode(self, z_quantized):
        return self.decoder(z_quantized)
    
    def forward(self, x):
        z = self.encode(x)
        z_q, vq_loss,_ = self.quantize(z)
        x_hat = self.decode(z_q)
        return x_hat, vq_loss
    
    def loss_function(self, x, x_hat, vq_loss):
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + vq_loss
        return loss
    
    def create_grid(self, val_loader, device, figsize=(10, 10), title=None):
        N = 9
        x, _ = next(iter(val_loader))
        x = x.to(device)
        # ecode and quantize
        z = self.encode(x)
        latents_shape = z.shape
        _,_,encoding_inds = self.quantize(z)

        #samp ind should be long Tensor
        samp_ind = torch.zeros((latents_shape[2]*latents_shape[3]*N, 1), dtype=torch.long).to(device)

        for i in range(samp_ind.shape[0]):
            ind = i#= np.random.randint(0, encoding_inds.shape[0])
            samp_ind[i] = encoding_inds[ind]
        encoding_one_hot = torch.zeros(samp_ind.size(0), self.num_embeddings, device=device)
        encoding_one_hot.scatter_(1, samp_ind, 1)  # [BHW x K]
        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.vq_layer.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view((N, latents_shape[2], latents_shape[3], latents_shape[1]))  # [B x H x W x D]

        # decode
        samples = self.decode(quantized_latents.permute(0, 3, 1, 2).contiguous()).detach().cpu()
        samples = (samples + 1) / 2

        fig = plt.figure(figsize=figsize)
        grid_size = int(np.sqrt(samples.shape[0]))
        grid = torchvision.utils.make_grid(samples, nrow=grid_size).permute(1, 2, 0)
        # save grid image
        plt.imshow(grid)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.savefig(os.path.join(figures_dir, f"VQVAE_{title}.png"))
        plt.close(fig)
        return grid
    
    def train_model(self, data_loader, epochs, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        epochs_bar = trange(epochs)
        for epoch in epochs_bar:
            acc_loss = 0.0
            for _, (x, _) in enumerate(data_loader):
                x = x.to(device)
                optimizer.zero_grad()
                x_hat, vq_loss = self.forward(x)
                loss = self.loss_function(x, x_hat, vq_loss)
                loss.backward()
                optimizer.step()
                acc_loss += loss.item()
            epochs_bar.set_description(f"Loss: {acc_loss/len(data_loader.dataset):.8f}")
            if epoch % 5 == 0:
                self.create_grid(data_loader, device, title=f"Epoch_{epoch}")
        torch.save(self.state_dict(), os.path.join(models_dir, "VQVAE.pt"))
        return self

    
