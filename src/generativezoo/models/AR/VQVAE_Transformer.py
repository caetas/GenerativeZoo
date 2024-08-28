####################################################################################################################################################################
### Code adapted from: https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_vqvae_transformer/2d_vqvae_transformer_tutorial.ipynb ###
####################################################################################################################################################################

import torch
import torch.nn as nn
from generative.inferers import VQVAETransformerInferer
from generative.networks.nets import VQVAE, DecoderOnlyTransformer
from generative.utils.ordering import Ordering
from generative.utils.enums import OrderingType
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import os
from config import models_dir
import wandb

def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, 'VQVAE_Transformer')):
        os.makedirs(os.path.join(models_dir, 'VQVAE_Transformer'))


class VQVAETransformer(nn.Module):
    def __init__(self, args, channels=3, img_size=32):
        '''
        VQVAETransformer model
        :param args: arguments
        :param channels: number of channels in the input image
        :param img_size: size of the input image
        '''
        super(VQVAETransformer, self).__init__()
        self.vqvae = VQVAE(spatial_dims=2,
                            in_channels=channels,
                            out_channels=channels,
                            num_res_layers=args.num_res_layers,
                            downsample_parameters=args.downsample_parameters,
                            upsample_parameters=args.upsample_parameters,
                            num_channels=args.num_channels,
                            num_res_channels=args.num_res_channels,
                            num_embeddings=args.num_embeddings,
                            embedding_dim=args.embedding_dim)
        
        test_input = torch.randn(1, channels, img_size, img_size)
        self.spatial_shape = self.vqvae.encode_stage_2_inputs(test_input).shape[2:]
        self.ordering = Ordering(ordering_type=OrderingType.RASTER_SCAN.value, spatial_dims=2, dimensions=(1,) + self.spatial_shape) # order the encoded features in raster scan order
        self.bos = args.num_embeddings  # Begin of Sentence (BOS) token
        
        self.transformer = DecoderOnlyTransformer(num_tokens=args.num_embeddings + 1,  # 256 from num_embeddings input of VQVAE + 1 for Begin of Sentence (BOS) token
                                                    max_seq_len=self.spatial_shape[0] * self.spatial_shape[1],
                                                    attn_layers_dim=args.attn_layers_dim,
                                                    attn_layers_depth=args.attn_layers_depth,
                                                    attn_layers_heads=args.attn_layers_heads,
                                                    )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vqvae.to(self.device)
        self.transformer.to(self.device)
        self.inferer = VQVAETransformerInferer()
        self.channels = channels
        self.img_size = img_size
        self.no_wandb = args.no_wandb
    
    def train_VQVAE(self, args, train_loader, verbose=True):
        '''
        Train the VQVAE model
        :param args: arguments
        :param train_loader: training data loader
        '''

        optimizer = torch.optim.Adam(params=self.vqvae.parameters(), lr=args.lr)
        l1_loss = nn.L1Loss()

        best_loss = np.inf

        epoch_bar = trange(args.n_epochs, desc='Epochs')
        for epoch in epoch_bar:
            self.vqvae.train()
            acc_loss = 0
            for x,_ in tqdm(train_loader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)
                optimizer.zero_grad()
                x_recon, quant_l, = self.vqvae(x)
                recon_loss = l1_loss(x_recon.float(), x.float())

                loss = recon_loss + quant_l

                loss.backward()
                optimizer.step()
                acc_loss += loss.item()*x.shape[0]

            acc_loss /= len(train_loader.dataset)
            if not self.no_wandb:
                wandb.log({'loss_vqvae': acc_loss})
            epoch_bar.set_postfix(loss=acc_loss)
            if acc_loss < best_loss:
                best_loss = acc_loss
                torch.save(self.vqvae.state_dict(), os.path.join(models_dir, 'VQVAE_Transformer', f'VQVAE_{args.dataset}.pt'))
            if (epoch+1) % args.sample_and_save_freq == 0 or epoch == 0:
                self.reconstruct(x[:8])

    def train_Transformer(self, args, train_loader, verbose=True):
        '''
        Train the Transformer model
        :param args: arguments
        :param train_loader: training data loader
        '''

        optimizer = torch.optim.Adam(params=self.transformer.parameters(), lr=args.lr_t)
        ce_loss = nn.CrossEntropyLoss()

        epoch_bar = trange(args.n_epochs_t, desc='Epochs')
        best_loss = np.inf

        for epoch in epoch_bar:
            self.transformer.train()
            acc_loss = 0
            for x,_ in tqdm(train_loader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)
                optimizer.zero_grad()
                logits, targets, _ = self.inferer(x, self.vqvae, self.transformer, self.ordering, return_latent=True)
                logits = logits.transpose(1, 2)

                loss = ce_loss(logits, targets)

                loss.backward()
                optimizer.step()
                acc_loss += loss.item()*x.shape[0]

            acc_loss /= len(train_loader.dataset)
            if not self.no_wandb:
                wandb.log({'loss_transformer': acc_loss})
            epoch_bar.set_postfix(loss=acc_loss)
            if acc_loss < best_loss:
                best_loss = acc_loss
                torch.save(self.transformer.state_dict(), os.path.join(models_dir, 'VQVAE_Transformer', f'Transformer_{args.dataset}.pt'))
            if (epoch+1) % args.sample_and_save_freq == 0 or epoch == 0:
                self.sample(16)
    
    def train_model(self, args, train_loader_a, train_loader_b, verbose=True):
        '''
        Train the VQVAE and Transformer models
        :param args: arguments
        :param train_loader_a: training data loader for VQVAE
        :param train_loader_b: training data loader for Transformer
        '''
        create_checkpoint_dir()
        print('Training VQVAE...')
        self.train_VQVAE(args, train_loader_a, verbose)
        # load the best VQVAE model
        self.vqvae.load_state_dict(torch.load(os.path.join(models_dir, 'VQVAE_Transformer', f'VQVAE_{args.dataset}.pt')))
        print('Training Transformer...')
        self.train_Transformer(args, train_loader_b, verbose)

    @torch.no_grad()
    def reconstruct(self, x, train=True):
        '''
        Reconstruct the input image
        :param x: input image
        :param train: whether to log the image to wandb
        '''
        self.vqvae.eval()
        x = x.to(self.device)
        x_recon, _ = self.vqvae(x)
        # clip the values to [0, 1]
        x_recon = torch.clamp(x_recon, 0, 1)
        # make a grid of images with original in top row and reconstructed in bottom row
        grid = make_grid(torch.cat((x, x_recon), dim=0), nrow=x.shape[0])
        # plot the grid
        fig = plt.figure(figsize=(16, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        if train:
            if not self.no_wandb:
                wandb.log({'reconstruction': fig})
        else:
            plt.show()
        plt.close(fig)
    
    @torch.no_grad()
    def sample(self, num_samples, train=True):
        '''
        Generate samples from the model
        :param num_samples: number of samples to generate
        :param train: whether to log the samples to wandb
        '''
        self.vqvae.eval()
        self.transformer.eval()
        images = []
        for _ in tqdm(range(num_samples)):
            sample = self.inferer.sample(transformer_model=self.transformer, vqvae_model=self.vqvae, ordering=self.ordering, latent_spatial_dim=(self.spatial_shape[0], self.spatial_shape[1]), starting_tokens=self.bos * torch.ones((1, 1), device=self.device), verbose=False)
            images.append(sample)
        images = torch.cat(images, dim=0)
        fig = plt.figure(figsize=(10, 10))
        grid = make_grid(images, nrow=int(num_samples**0.5))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        if train:
            if not self.no_wandb:
                wandb.log({'samples': fig})
        else:
            plt.show()

    def load_checkpoint(self, checkpoint_vqvae=None, checkpoint_transformer=None):
        '''
        Load the model checkpoints
        :param checkpoint_vqvae: checkpoint for VQVAE model
        :param checkpoint_transformer: checkpoint for Transformer model
        '''
        if checkpoint_vqvae is not None:
            self.vqvae.load_state_dict(torch.load(checkpoint_vqvae))
        if checkpoint_transformer is not None:
            self.transformer.load_state_dict(torch.load(checkpoint_transformer))




                

                           