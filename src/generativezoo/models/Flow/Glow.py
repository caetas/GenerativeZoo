###############################################################
### Code adapted from https://github.com/y0ast/Glow-PyTorch ###
###############################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np
import os
from config import models_dir
from sklearn.metrics import roc_auc_score


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
    
def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    if temperature == None:
        temperature = 1
    z = torch.normal(mean, torch.exp(logs) * temperature)

    return z


def squeeze2d(input, factor):
    """
    squeeze a 4D tensor of B x C x H x W into a 4D tensor of B x (C*f*f) x (H/f) x (W/f)
    """
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(input, factor):
    """
    unsqueeze a 4D tensor of B x C x H x W into a 4D tensor of B x (C/f/f) x (H*f) x (W*f)
    """
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


class _ActNorm(nn.Module):
    def __init__(self, num_features, scale=1.0):
        """
        Activation Normalization
        Initialize the bias and scale with a given minibatch,
        so that the output per-channel have zero mean and unit variance for that.

        After initialization, `bias` and `logs` will be trained as parameters.
        params:
            num_features: int, number of features in the input tensor
            scale: float, scale factor for the initialized logs
        """
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        """
        Initialize the parameters with the first minibatch
        """
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        """
        input: 4D Tensor, BCHW
        logdet: 1D Tensor, batch size
        reverse: bool, if True, reverse the transformation
        """
        self._check_input_dim(input)

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()
            )
        )


class LinearZeros(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor

        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1, dtype=torch.long)
        self.indices_inverse = torch.zeros((self.num_channels), dtype=torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4

        if not reverse:
            input = input[:, self.indices, :, :]
            return input
        else:
            return input[:, self.indices_inverse, :, :]


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape),'reduced')[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.linalg.lu_factor(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(self.lower.device)
            self.eye = self.eye.to(self.lower.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous().to(self.upper.device)
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1).to(input.device), dlogdet.to(input.device)

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, actnorm_scale, flow_permutation, flow_coupling, LU_decomposed):
        super().__init__()
        self.flow_coupling = flow_coupling

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, actnorm_scale, flow_permutation, flow_coupling, LU_decomposed):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                    )
                )
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z
    
def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'Glow')):
    os.makedirs(os.path.join(models_dir, 'Glow'))


class Glow(nn.Module):
    def __init__(self,image_shape,hidden_channels,args):
        '''
        Glow model
        :param image_shape: tuple, shape of the input image
        :param hidden_channels: int, number of hidden channels
        :param args: arguments
        '''
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = FlowNet(image_shape=image_shape,hidden_channels=hidden_channels,K=args.K,L=args.L,actnorm_scale=args.actnorm_scale,flow_permutation=args.flow_permutation,flow_coupling=args.flow_coupling,LU_decomposed=args.LU_decomposed).to(self.device)
        self.y_condition = args.y_condition
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.y_classes = args.num_classes
        self.n_bits = args.n_bits
        self.sample_and_save_freq = args.sample_and_save_freq
        self.K = args.K
        self.L = args.L
        self.hidden_channels = args.hidden_channels
        self.dataset = args.dataset
        self.learn_top = args.learn_top
        self.y_condition = args.y_condition
        self.no_wandb = args.no_wandb

        # learned prior
        if self.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if self.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(args.num_classes, 2 * C)
            self.project_class = LinearZeros(C, args.num_classes)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )

        self=self.to(self.device)

    def prior(self, data, y_onehot=None, n=16):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(n, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x, y_onehot)
        mean = mean.to(x.device)
        logs = logs.to(x.device)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits
    
    @torch.no_grad()
    def reverse_flow(self, z, y_onehot, temperature, n=16):
        if z is None:
            mean, logs = self.prior(z, y_onehot, n=n)
            z = gaussian_sample(mean, logs, temperature)
        x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
    
    def preprocess(self, x):
        x = x * 255  # undo ToTensor scaling to [0,1]

        n_bins = 2 ** self.n_bits
        if self.n_bits < 8:
            x = torch.floor(x / 2 ** (8 - self.n_bits))
        x = x / n_bins - 0.5

        return x

    def train_model(self,dataloader, args):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        epoch_bar = trange(self.n_epochs, desc="Epochs")
        self.train()
        create_checkpoint_dir()
        best_loss = np.inf

        for epoch in epoch_bar:
            total_loss = 0
            for x, label in tqdm(dataloader, desc = "Batches", leave = False):
                x = self.preprocess(x)
                x = x.to(self.device)
                optimizer.zero_grad()
                if self.y_condition:
                    label_one_hot = F.one_hot(label, self.y_classes).to(self.device)
                    z, nll, y_logits = self.forward(x, label_one_hot)
                    losses = compute_loss_y(nll, y_logits, 0.01, label_one_hot, True)
                else:
                    z, nll, y_logits = self.forward(x, None)
                    losses = compute_loss(nll)
                losses["total_loss"].backward()

                if args.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.parameters(), args.max_grad_clip)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                optimizer.step()
                total_loss += losses["total_loss"].item()

            epoch_bar.set_postfix(loss=total_loss/len(dataloader))
            if self.no_wandb:
                wandb.log({'Loss': total_loss/len(dataloader)})

            if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:
                self.sample()
            
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.state_dict(), os.path.join(models_dir,'Glow', f"Glow_{self.dataset}_{self.K}_{self.L}_{self.hidden_channels}.pt"))
                self.load_state_dict(torch.load(os.path.join(models_dir,'Glow', f"Glow_{self.dataset}_{self.K}_{self.L}_{self.hidden_channels}.pt")))

    
    def sample(self, n=16, y_onehot=None, temperature=None, train=True):
        #z = torch.randn(n, *self.flow.output_shapes[-1]).to(self.device)
        x = self.reverse_flow(None, y_onehot, temperature, n=n)
        x = torch.clamp(x, -0.5, 0.5)
        x += 0.5
        # plot torch grid
        grid = make_grid(x, nrow=int(n ** 0.5), padding=0)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis("off")
        if train:
            if self.no_wandb:
                wandb.log({'Samples': fig})
        else:
            plt.show()
        plt.close(fig)

    @torch.no_grad()
    def outlier_detection(self, in_loader, out_loader):

        in_scores = []
        out_scores = []

        for x, _ in tqdm(in_loader, desc="In-distribution"):
            x = self.preprocess(x)
            x = x.to(self.device)
            z, nll, y_logits = self.forward(x, None)
            losses = compute_loss(nll, reduction="none")
            in_scores.extend(losses["nll"].detach().cpu().numpy())

        for x, _ in tqdm(out_loader, desc="Out-of-distribution"):
            x = self.preprocess(x)
            x = x.to(self.device)
            z, nll, y_logits = self.forward(x, None)
            losses = compute_loss(nll, reduction="none")
            out_scores.extend(losses["nll"].detach().cpu().numpy())
        
        # get auc
        in_scores = np.array(in_scores)
        out_scores = np.array(out_scores)
        in_labels = np.zeros_like(in_scores)
        out_labels = np.ones_like(out_scores)
        scores = np.concatenate([in_scores, out_scores])
        labels = np.concatenate([in_labels, out_labels])
        auc = roc_auc_score(labels, scores)
        # print with 4 decimal places
        print(f"AUC: {auc:.4f}")
        
        # plot histograms of scores in same plot
        plt.hist(in_scores, bins=100, alpha=0.5, label='In-distribution')
        plt.hist(out_scores, bins=100, alpha=0.5, label='Out-of-distribution')
        plt.legend(loc='upper right')
        plt.xlabel('NLL')
        plt.ylabel('Frequency')
        plt.show()
        return in_scores, out_scores
    
    def load_checkpoint(self, args):
        if args.checkpoint is not None:
            self.load_state_dict(torch.load(args.checkpoint))
        self.set_actnorm_init()

def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses