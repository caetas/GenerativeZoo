###########################################################
### Code based on: https://github.com/cloneofsimo/minRF ###
###########################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import wandb
import os
from config import models_dir
import copy
from collections import OrderedDict
from diffusers.models import AutoencoderKL
from accelerate import Accelerator

@torch.no_grad()
def update_ema(ema_model, model, decay=0.5):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # if name contains "module" then remove module
        if "module" in name:
            name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = t.to(x.dtype) + y.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=256, n_layers=16, n_heads=32, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)


def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, "RectifiedFlows")):
        os.makedirs(os.path.join(models_dir, "RectifiedFlows"))

class RF(nn.Module):

    def __init__(self, args, img_size, channels, ln=True):
        super(RF, self).__init__()
        '''
        Initialize the model
        :param args: argparse.ArgumentParser, arguments
        :param img_size: int, size of the image
        :param channels: int, number of channels in the image
        :param ln: bool, whether to use layer normalization
        '''
        self.args = args
        self.conditional = args.conditional
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(self.device) if args.latent else None
        self.channels = channels
        self.img_size = img_size

        # If using VAE, change the number of channels and image size accordingly
        if self.vae is not None:
            self.channels = 4
            self.img_size = self.img_size // 8

        if self.conditional:
            self.model = DiT_Llama(self.channels, self.img_size, args.patch_size, args.dim, args.n_layers, args.n_heads, args.multiple_of, args.ffn_dim_multiplier, args.norm_eps, args.class_dropout_prob, args.num_classes)
        else:
            self.model = DiT_Llama(self.channels, self.img_size, args.patch_size, args.dim, args.n_layers, args.n_heads, args.multiple_of, args.ffn_dim_multiplier, args.norm_eps, 0, 1)
        self.ln = ln
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.num_classes = args.num_classes
        self.sample_and_save_freq = args.sample_and_save_freq
        self.dataset = args.dataset
        self.sample_steps = args.sample_steps
        self.cfg = args.cfg
        self.warmup = args.warmup
        self.decay = args.decay
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {model_size}, {model_size / 1e6}M")
        self.model.to(self.device)
        self.no_wandb = args.no_wandb
        self.snapshot = args.n_epochs//args.snapshots
        if args.train:
            self.ema = copy.deepcopy(self.model)
            self.ema_rate = args.ema_rate
            for param in self.ema.parameters():
                param.requires_grad = False

    def forward(self, x, cond):
        '''
        Forward pass through the model
        :param x: torch.Tensor, input image
        :param cond: torch.Tensor, class labels
        '''

        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss
    
    @torch.no_grad()
    def encode(self, x):
        '''
        Encode the input image
        :param x: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.encode(x)
        else:
            return self.vae.encode(x)
        
    @torch.no_grad()    
    def decode(self, z):
        '''
        Decode the input image
        :param z: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.decode(z)
        else:
            return self.vae.decode(z)

    @torch.no_grad()
    def get_sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, train=False, accelerate=None):
        '''
        Generate samples from the model
        :param z: torch.Tensor, random noise
        :param cond: torch.Tensor, class labels
        :param null_cond: torch.Tensor, class labels for unconditional samples
        :param sample_steps: int, number of steps to sample
        :param cfg: float, conditioning factor
        :param train: bool, whether to log samples to wandb
        '''
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        if self.conditional:
            cond = torch.cat([cond, null_cond], dim=0)

        for i in tqdm(range(sample_steps, 0, -1), desc='Sampling', leave=False):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            if self.conditional:
                z = z.repeat(2, 1, 1, 1)
                t = t.repeat(2)
                if train:
                    v = self.ema(z, t, cond.long().to(z.device))
                else:
                    v = self.model(z, t, cond.long().to(z.device))
                vc = v[:b]
                vu = v[b:]
                vc = vu + cfg * (vc - vu)
                z = z[:b]
            
            else :
                if train:
                    vc = self.ema(z, t, torch.zeros_like(cond).long().to(z.device))
                else:
                    vc = self.model(z, t, torch.zeros_like(cond).long().to(z.device))

            z = z - dt * vc
            images.append(z)
        
        imgs = images[-1]

        if self.vae is not None:
            imgs = self.decode(imgs / 0.18215).sample

        imgs = imgs*0.5 + 0.5
        imgs = imgs.clamp(0, 1)
        grid = make_grid(imgs, nrow=4)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        if train:
            if not self.no_wandb:
                accelerate.log({"samples": fig})
        else:   
            plt.show()
        plt.close(fig)
    
    def train_model(self, train_loader, verbose=True):
        '''
        Train the model
        :param train_loader: PyTorch DataLoader object
        :param verbose: bool, whether to display progress bar
        '''
        accelerate = Accelerator(log_with="wandb")
        if not self.no_wandb:
            accelerate.init_trackers(project_name='RectifiedFlows',
            config = {
                        "dataset": self.args.dataset,
                        "batch_size": self.args.batch_size,
                        "n_epochs": self.args.n_epochs,
                        "lr": self.args.lr,
                        "patch_size": self.args.patch_size,
                        "dim": self.args.dim,
                        "n_layers": self.args.n_layers,
                        "n_heads": self.args.n_heads,
                        "multiple_of": self.args.multiple_of,
                        "ffn_dim_multiplier": self.args.ffn_dim_multiplier,
                        "norm_eps": self.args.norm_eps,
                        "class_dropout_prob": self.args.class_dropout_prob,
                        "num_classes": self.args.num_classes,
                        "conditional": self.args.conditional,
                        "ema_rate": self.args.ema_rate,
                        "warmup": self.args.warmup,
                        "latent": self.args.latent,
                        "decay": self.args.decay,
                        "size": self.args.size,       
                },
                init_kwargs={"wandb":{"name": f"RectifiedFlows_{self.args.dataset}"}})
        
        create_checkpoint_dir()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.n_epochs*len(train_loader), pct_start=self.warmup/self.n_epochs, anneal_strategy='cos', cycle_momentum=False, div_factor=self.lr/1e-6, final_div_factor=1)

        epoch_bar = trange(self.n_epochs, desc="Epochs")

        if self.vae is None:
            train_loader, self.model, optimizer, scheduler, self.ema = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema)
        else:
            train_loader, self.model, optimizer, scheduler, self.ema, self.vae = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema, self.vae)

        update_ema(self.ema, self.model, 0)

        best_loss = float("inf")
        for epoch in epoch_bar:
            self.model.train()
            train_loss = 0
            for (x, cond) in tqdm(train_loader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)
                cond = cond.to(self.device)

                with accelerate.autocast():

                    if self.vae is not None:
                        with torch.no_grad():
                            # if x has one channel, make it 3 channels
                            if x.shape[1] == 1:
                                x = torch.cat((x, x, x), dim=1)
                            x = self.encode(x).latent_dist.sample().mul_(0.18215)

                    optimizer.zero_grad()

                    if self.conditional:
                        loss, _ = self.forward(x, cond)
                    else:
                        loss, _ = self.forward(x, torch.zeros_like(cond).long())

                    accelerate.backward(loss)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()*x.shape[0]
                update_ema(self.ema, self.model, self.ema_rate)

            accelerate.wait_for_everyone()
            
            if not self.no_wandb:
                accelerate.log({"train_loss": train_loss / len(train_loader.dataset)})
                accelerate.log({"lr": scheduler.get_last_lr()[0]})
                
            epoch_bar.set_postfix(loss=train_loss / len(train_loader.dataset))

            if train_loss/len(train_loader.dataset) < best_loss:
                best_loss = train_loss/len(train_loader.dataset)

            if (epoch+1) % self.snapshot == 0:
                ema_to_save = accelerate.unwrap_model(self.ema)
                accelerate.save(ema_to_save.state_dict(), os.path.join(models_dir, "RectifiedFlows", f"{'Lat' if self.vae is not None else ''}{'CondRF' if self.conditional else 'RF'}_{self.dataset}_epoch{epoch+1}.pt"))
        
            if epoch == 0 or ((epoch+1) % self.sample_and_save_freq == 0):
                cond = torch.arange(0, 16).cuda() % self.num_classes
                z = torch.randn(16, self.channels, self.img_size, self.img_size).to(self.device)
                null_cond = self.num_classes*torch.ones_like(cond).long() if self.conditional else torch.zeros_like(cond).long()
                self.get_sample(z, cond, train=True, sample_steps=self.sample_steps, cfg=self.cfg, null_cond=null_cond, accelerate=accelerate)
        
        accelerate.end_training()

    @torch.no_grad()
    def sample(self, num_samples):
        '''
        Generate samples from the model
        :param num_samples: int, number of samples to generate
        '''
        self.model.eval()
        cond = torch.arange(0, num_samples).cuda() % self.num_classes
        z = torch.randn(num_samples, self.channels, self.img_size, self.img_size).to(self.device)
        null_cond = self.num_classes*torch.ones_like(cond).long() if self.conditional else torch.zeros_like(cond).long()
        self.get_sample(z, cond, train=False, sample_steps=self.sample_steps, cfg=self.cfg, null_cond=null_cond)

    def load_checkpoint(self, checkpoint):
        '''
        Load a model checkpoint
        :param checkpoint: str, path to the checkpoint
        '''
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))