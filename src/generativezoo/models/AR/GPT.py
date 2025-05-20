#######################################################################
######### Code based on: https://github.com/karpathy/nanoGPT ##########
#######################################################################

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from ..GAN.VQGAN import VQModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from accelerate import Accelerator
import os
from config import models_dir

def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    checkpoint_dir = os.path.join(models_dir, "GPT")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embed_dim_t % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim_t, 3 * config.embed_dim_t, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim_t, config.embed_dim_t, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout_t)
        self.resid_dropout = nn.Dropout(config.dropout_t)
        self.n_head = config.n_head
        self.embed_dim_t = config.embed_dim_t
        self.dropout = config.dropout_t
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim_t, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.embed_dim_t, 4 * config.embed_dim_t, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.embed_dim_t, config.embed_dim_t, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout_t)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.embed_dim_t, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embed_dim_t, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, args, channels, input_size):
        super().__init__()
        assert args.n_embed is not None
        assert args.block_size is not None
        self.args = args

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.n_embed + 1, args.embed_dim_t),
            wpe = nn.Embedding(args.block_size, args.embed_dim_t),
            drop = nn.Dropout(args.dropout_t),
            h = nn.ModuleList([Block(args) for _ in range(args.n_layer)]),
            ln_f = LayerNorm(args.embed_dim_t, bias=args.bias),
        ))
        self.lm_head = nn.Linear(args.embed_dim_t, args.n_embed+1, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"Cannot forward sequence of length {t}, block size is only {self.args.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, embed_dim)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, embed_dim)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.args.block_size
        self.args.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.args.block_size else idx[:, -self.args.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # logit for bos token should be -inf
            logits[:, self.args.n_embed] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
class VQGAN_GPT(nn.Module):
    def __init__(self, args, channels, input_size):
        super().__init__()
        self.VAE = VQModel(args, channels, input_size)
        self.GPT = GPT(args, channels, input_size)
        self.VAE.load_checkpoint(args.checkpoint_vae)
        self.zshape = (args.num_samples, args.z_channels, input_size//(2**(len(args.ch_mult)-1)), input_size//(2**(len(args.ch_mult)-1)))
        args.block_size = self.zshape[2] * self.zshape[3]
        self.args = args
        for param in self.VAE.parameters():
            param.requires_grad = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.lr = args.lr
        self.resolution = input_size

    def load_checkpoint(self, checkpoint):
        """
        Load the model checkpoint.
        """
        if checkpoint is not None:
            if os.path.isfile(checkpoint):
                print(f"Loading checkpoint '{checkpoint}'")
                self.GPT.load_state_dict(torch.load(checkpoint, weights_only=False))


    def train_model(self, train_loader, val_loader):
        """
        Train the model using the provided data loaders.
        """
        create_checkpoint_dir()

        accelerate = Accelerator(log_with='wandb')

        accelerate.init_trackers("GPT", config={
            "lr": self.lr,
            "batch_size": self.args.batch_size,
            "n_epochs": self.args.n_epochs,
            "n_embed": self.args.n_embed,
            "embed_dim": self.args.embed_dim_t,
            "resolution": self.resolution,
            "n_layer": self.args.n_layer,
            "n_head": self.args.n_head,
            "block_size": self.args.block_size,
            "dropout_t": self.args.dropout_t,
            "bias": self.args.bias,
            "weight_decay": self.args.weight_decay,
            "embed_dim_t": self.args.embed_dim_t,
            "betas": self.args.betas,
            "dataset": self.args.dataset,
        },
        init_kwargs={"wandb":{"name": f"GPT_{self.args.dataset}"}})

        optimizer = self.GPT.configure_optimizers(self.args.weight_decay, self.args.lr, self.args.betas, self.device)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, total_steps=self.args.n_epochs*len(train_loader), pct_start=0.1, anneal_strategy='cos', cycle_momentum=False, div_factor=self.lr/1e-6, final_div_factor=1)

        # Move model and optimizer to the accelerator
        self.GPT, self.VAE, optimizer, scheduler, train_loader, val_loader = accelerate.prepare(
            self.GPT, self.VAE, optimizer, scheduler, train_loader, val_loader
        )   

        #iterate over the training data
        for epoch in tqdm(range(self.args.n_epochs), desc="Training Epochs"):
            self.GPT.train()
            epoch_loss = 0
            for batch,_ in tqdm(train_loader, desc="Training Batches", leave=False):
                batch = batch.to(self.device)
                encoded, y = self.encode(batch)
                # x should be n-1 elements of y and append n_embed at the beginning
                x = torch.cat((torch.full((encoded.shape[0],1), self.args.n_embed).to(self.device), y[:,:-1]), dim=1)
                # forward pass
                logits, loss = self.GPT(x, targets=y)
                # backward pass
                optimizer.zero_grad()
                accelerate.backward(loss)
                optimizer.step()
                epoch_loss += loss.item()*len(batch)
                scheduler.step()

            epoch_loss /= len(train_loader.dataset)
            accelerate.log({"epoch_loss": epoch_loss})
            # Validation step
            # generate some samples
            if (epoch+1) % self.args.sample_and_save_freq == 0:
                self.GPT.eval()
                self.sample(train=True, accelerate=accelerate)

                model_to_save = accelerate.unwrap_model(self.GPT)
                # Save the model
                accelerate.save(model_to_save.state_dict(), os.path.join(models_dir, "GPT", f"gpt_{self.args.dataset}_{epoch+1}.pt"))
                epoch_loss = 0
                with torch.no_grad():
                    for batch,_ in tqdm(val_loader, desc="Validation Batches", leave=False):
                        batch = batch.to(self.device)
                        encoded, y = self.encode(batch)
                        # x should be n-1 elements of y and append n_embed at the beginning
                        x = torch.cat((torch.full((encoded.shape[0],1), self.args.n_embed).to(self.device), y[:,:-1]), dim=1)
                        # forward pass
                        logits, loss = self.GPT(x, targets=y)
                        # backward pass
                        epoch_loss += loss.item()*len(batch)
                    epoch_loss /= len(val_loader.dataset)
                    accelerate.log({"val_loss": epoch_loss})
        
    @torch.no_grad()
    def sample(self, train=False, accelerate=None):
        """
        Sample from the model.
        """
        # init token is just a single token with value n_embed
        idx = torch.full((self.args.num_samples,1), self.args.n_embed).to(self.device)
        # generate some samples
        samples = self.GPT.generate(idx, 64, temperature=self.args.temperature, top_k=self.args.top_k)[:, 1:]
        decoded = self.decode(samples, self.zshape)
        decoded = decoded *0.5 + 0.5
        decoded = decoded.clamp(0, 1)
        # plot the samples
        grid = make_grid(decoded, nrow=4, normalize=True)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        if train:
            accelerate.log({"sample": fig})
        else:
            plt.show()
        plt.close(fig)

    @torch.no_grad()
    def encode(self, batch):
        """
        Encode the input batch using the VAE encoder.
        """
        quant_z, _, info = self.VAE.encode(batch)
        indices = info[2].view(quant_z.size(0), -1)
        # S-pattern transform
        indices = self.s_pattern_transform(indices, quant_z.size(2), quant_z.size(3))

        return quant_z, indices
    
    @torch.no_grad()
    def decode(self, indices, zshape):
        """
        Decode the input indices using the VAE decoder.
        """
        # S-pattern transform
        indices = self.s_pattern_transform(indices, zshape[2], zshape[3], inverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.VAE.quantize.get_codebook_entry(indices.reshape(-1), shape=bhwc)
        x = self.VAE.decode(quant_z)
        #x= self.VAE.decode_code(indices.reshape(-1))
        
        return x
    
    @torch.no_grad()
    def s_pattern_transform(self, tokens: torch.Tensor, height: int, width: int, inverse: bool = False) -> torch.Tensor:
        """
        Transform tokens between raster and S-pattern (boustrophedon) order.

        Args:
            tokens (torch.Tensor): Tensor of shape (B, H*W) or (H*W,)
            height (int): Grid height
            width (int): Grid width
            mode (str): "reorder" to convert to S-pattern,
                        "restore" to revert back to raster order

        Returns:
            torch.Tensor: Transformed tokens in same shape as input
        """
        # Create base S-pattern indices
        indices = torch.arange(height * width).reshape(height, width)
        for i in range(height):
            if i % 2 == 1:
                indices[i] = indices[i].flip(0)
        s_indices = indices.flatten()

        # Compute reverse mapping
        if inverse:
            mapping = torch.empty_like(s_indices)
            mapping[s_indices] = torch.arange(height * width)
        else:
            mapping = s_indices

        # Apply transformation
        if tokens.dim() == 2:
            return tokens[:, mapping]
        elif tokens.dim() == 1:
            return tokens[mapping]
        else:
            raise ValueError("tokens must be of shape (B, H*W) or (H*W,)")