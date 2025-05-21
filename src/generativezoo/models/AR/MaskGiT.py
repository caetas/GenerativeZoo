# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
from torch import nn
import os
import random
import time
import math

import numpy as np
from tqdm import tqdm
from collections import deque

import torch.nn.functional as F
import torchvision.utils as vutils
from ..GAN.VQGAN import VQModel
from matplotlib import pyplot as plt

from accelerate import Accelerator
from config import models_dir

def create_checkpoint_dir():
    """ Create the checkpoint directory if it does not exist """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, "MaskGiT")):
        os.makedirs(os.path.join(models_dir, "MaskGiT"))


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """ Initialize the Attention module.
            :param:
                embed_dim     -> int : Dimension of the embedding
                num_heads     -> int : Number of heads
                dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = x + attention_value
            x = x + ff(x)
            l_attn.append(attention_weight)
        return x, l_attn


class MaskTransformer(nn.Module):
    def __init__(self, img_size=256, hidden_dim=768, codebook_size=1024, f_factor=16, depth=24, heads=8, mlp_dim=3072, dropout=0.1, nclass=1000):
        """ Initialize the Transformer model.
            :param:
                img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                nclass         -> int:     Number of classes (default: 1000)
        """

        super().__init__()
        self.nclass = nclass
        self.patch_size = f_factor
        self.codebook_size = codebook_size
        self.tok_emb = nn.Embedding(codebook_size+1+nclass+1, hidden_dim)  # +1 for the mask of the viz token, +1 for mask of the class
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, (self.patch_size*self.patch_size)+1, hidden_dim)), 0., 0.02)

        # First layer before the Transformer block
        self.first_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.transformer = TransformerEncoder(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=1e-12),
        )

        # Bias for the last linear output
        self.bias = nn.Parameter(torch.zeros((self.patch_size*self.patch_size)+1, codebook_size+1+nclass+1))

    def forward(self, img_token, y=None, drop_label=None, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, w, h = img_token.size()

        cls_token = y.view(b, -1) + self.codebook_size + 1  # Shift the class token by the amount of codebook

        cls_token[drop_label] = self.codebook_size + 1 + self.nclass  # Drop condition
        input = torch.cat([img_token.view(b, -1), cls_token.view(b, -1)], -1)  # concat visual tokens and class tokens
        tok_embeddings = self.tok_emb(input)

        # Position embedding
        pos_embeddings = self.pos_emb
        x = tok_embeddings + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x)
        x = self.last_layer(x)

        logit = torch.matmul(x, self.tok_emb.weight.T) + self.bias   # Shared layer with the embedding

        if return_attn:  # return list of attention
            return logit[:, :self.patch_size * self.patch_size, :self.codebook_size + 1], attn

        return logit[:, :self.patch_size*self.patch_size, :self.codebook_size+1]
    
class MaskGIT(nn.Module):
    """ Masked Generative Image Transformer (MaskGIT) model
        :param
            args -> Namespace: Arguments for the model
    """

    def __init__(self, args, channels, img_size):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__()
        self.args = args                                                        # Main argument see main.py
        self.ae = VQModel(args, channels, img_size)
        self.ae.load_checkpoint(self.args.checkpoint_vae)                      # Load VQGAN
        self.codebook_size = self.args.n_embed   
        self.patch_size = img_size // (2**(len(args.ch_mult)-1))     # Load VQGAN
        print(f"Acquired codebook size: {self.codebook_size}, f_factor: {(2**(len(args.ch_mult)-1))}")
        self.vit = MaskTransformer(img_size=img_size, hidden_dim=args.hidden_dim, codebook_size=self.codebook_size, f_factor=self.patch_size, depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim, dropout=args.dropout_t, nclass=self.args.n_classes)                                    # Load Masked Bidirectional Transformer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)    # Get cross entropy loss
        self.optim = torch.optim.AdamW(self.vit.parameters(), lr=self.args.lr, betas=(0.9, 0.96), weight_decay=self.args.weight_decay)  # Get Adam Optimizer with weight decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get device
        self.vit.to(self.device)  # Send model to device
        self.ae.to(self.device)  # Send model to device
        self.args.mask_value = self.codebook_size  # Mask value for the maskGit
        self.num_samples = self.args.num_samples  # Number of samples to generate

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None, codebook_size=256):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

        return mask_code, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def train_model(self, train_loader, val_loader):
        """ Train the model """

        create_checkpoint_dir()  # Create the checkpoint directory if it does not exist

        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, self.args.lr, total_steps=self.args.n_epochs*len(train_loader), pct_start=0.1, anneal_strategy='cos', cycle_momentum=False, div_factor=self.args.lr/1e-6, final_div_factor=1)

        accelerate = Accelerator(log_with='wandb')

        accelerate.init_trackers("MaskGIT", config=self.args, init_kwargs={"wandb":{"name": f"MaskGiT_{self.args.dataset}"}})

        # Send model to accelerator
        self.vit, self.optim, scheduler, self.ae, train_loader, val_loader = accelerate.prepare(self.vit, self.optim, scheduler, self.ae, train_loader, val_loader)
        
        for epoch in tqdm(range(self.args.n_epochs), desc="Epoch", leave=True):
            epoch_loss = 0.
            self.vit.train()
            
            for batch, label in tqdm(train_loader, desc="Batch", leave=False):
                batch = batch.to(self.device)
                label = label.to(self.device)
                # Drop xx% of the condition for cfg
                drop_label = (torch.rand(batch.size(0)) < self.args.drop_label).bool().to(self.device)

                # VQGAN encoding to img tokens
                with torch.no_grad():
                    emb, _, [_, _, code] = self.ae.encode(batch)
                    code = code.reshape(batch.size(0), self.patch_size, self.patch_size)

                # Mask the encoded tokens
                masked_code, mask = self.get_mask_code(code, value=self.codebook_size, codebook_size=self.codebook_size)

                pred = self.vit(masked_code, label, drop_label=drop_label)  # The unmasked tokens prediction
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1))

                self.optim.zero_grad()

                accelerate.backward(loss)  # rescale to get more precise loss

                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient

                self.optim.step()  # Update the weights
                scheduler.step()  # Update the learning rate
                epoch_loss += loss.item()* batch.size(0)



            epoch_loss /= len(train_loader.dataset)
            accelerate.log({"train_loss": epoch_loss})

            if (epoch+1) % self.args.sample_and_save_freq == 0:

                # Save the model
                model_to_save = accelerate.unwrap_model(self.vit)
                accelerate.save(model_to_save.state_dict(), os.path.join(models_dir, "MaskGiT", f"MaskGIT_{self.args.dataset}_{epoch+1}.pth"))

                self.vit.eval()
                with torch.no_grad():
                    val_loss = 0.
                    for batch, label in tqdm(val_loader, desc="Batch", leave=False):
                        batch = batch.to(self.device)
                        label = label.to(self.device)
                        # Drop xx% of the condition for cfg
                        drop_label = (torch.rand(batch.size(0)) < self.args.drop_label).bool().to(self.device)

                        # VQGAN encoding to img tokens
                        emb, _, [_, _, code] = self.ae.encode(batch)
                        code = code.reshape(batch.size(0), self.patch_size, self.patch_size)

                        # Mask the encoded tokens
                        masked_code, mask = self.get_mask_code(code, value=self.codebook_size, codebook_size=self.codebook_size)

                        pred = self.vit(masked_code, label, drop_label=drop_label)

                        # Cross-entropy loss
                        loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1))
                        val_loss += loss.item()* batch.size(0)
                    val_loss /= len(val_loader.dataset)
                    accelerate.log({"val_loss": val_loss})


                gen_sample = self.get_sample(init_code=None,
                                            nb_sample=16,
                                            labels=None,
                                            sm_temp=self.args.sm_temp,
                                            w=self.args.cfg_w,
                                            randomize="linear",
                                            r_temp=self.args.r_temp,
                                            sched_mode=self.args.sched_mode,
                                            step=self.args.step
                                            )[0]
                fig = plt.figure(figsize=(8, 8))
                grid = vutils.make_grid(gen_sample, nrow=4, padding=0)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                accelerate.log({"sample": fig})
            
    def reco(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                # Decoding reel code
                _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
                if mask is not None:
                    # Decoding reel code with mask to hide
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask, (self.args.img_size, self.args.img_size)).to(self.device))
                    l_visual.append(__x2)
            if masked_code is not None:
                # Decoding masked code
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0,  self.codebook_size-1))
                l_visual.append(__x)

            if unmasked_code is not None:
                # Decoding predicted code
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
                l_visual.append(___x)

        return torch.cat(l_visual, dim=0)
    
    @torch.no_grad()
    def decode(self, indices, zshape):
        """
        Decode the input indices using the VAE decoder.
        """
        # S-pattern transform
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.ae.quantize.get_codebook_entry(indices.reshape(-1), shape=bhwc)
        x = self.ae.decode(quant_z)
        #x= self.VAE.decode_code(indices.reshape(-1))
        
        return x
    
    def load_checkpoint(self, checkpoint_path):
        """ Load the model checkpoint
            :param
            checkpoint_path -> str: path to the checkpoint
        """
        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                self.vit.load_state_dict(torch.load(checkpoint_path, weights_only=False))
                print(f"Checkpoint {checkpoint_path} loaded")
        else:
            print(f"Checkpoint {checkpoint_path} not found")
    
    @torch.no_grad()
    def sample(self):
        """ Sample the model
            :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        samples = self.get_sample(init_code=None,
                               nb_sample=self.num_samples,
                               labels=None,
                               sm_temp=self.args.sm_temp,
                               w=self.args.cfg_w,
                               randomize="linear",
                               r_temp=self.args.r_temp,
                               sched_mode=self.args.sched_mode,
                               step=self.args.step)[0]
        
        # Decode the generated code
        samples = self.decode(samples, zshape=(self.num_samples, self.args.z_channels, self.patch_size, self.patch_size))
        samples = samples * 0.5 + 0.5
        samples = samples.clamp(0, 1)
        grid = vutils.make_grid(samples, nrow=int(np.sqrt(self.num_samples)), padding=0)
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()
        plt.close(fig)


    def get_sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = torch.arange(0, nb_sample) % self.args.n_classes
                labels = torch.LongTensor(labels).to(self.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                if w != 0:
                    # Model Prediction
                    logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                        torch.cat([labels, labels], dim=0),
                                        torch.cat([~drop, drop], dim=0))
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    _w = w * (indice / (len(scheduler)-1))
                    # Classifier Free Guidance
                    logit = (1 + _w) * logit_c - _w * logit_u
                else:
                    logit = self.vit(code.clone(), labels, drop_label=~drop)

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf.squeeze()) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf.squeeze())

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            # decode the final prediction
            _code = torch.clamp(code, 0,  self.codebook_size-1)
            x = self.decode(_code, zshape=(nb_sample, self.args.z_channels, self.patch_size, self.patch_size))
            x = x* 0.5 + 0.5
            x = x.clamp(0, 1)

        self.vit.train()
        return x, l_codes, l_mask