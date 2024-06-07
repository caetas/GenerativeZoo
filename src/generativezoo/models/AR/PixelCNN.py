#########################################################################################################################################################
### Code adapted from: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.ipynb ###
#########################################################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
import os
from config import models_dir

class MaskedConvolution(nn.Module):
    
    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically 
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)
        
        # Mask as buffer => it is no parameter but still a tensor of the module 
        # (must be moved with the devices)
        self.register_buffer('mask', mask[None,None])
        
    def forward(self, x):
        self.conv.weight.data *= self.mask # Ensures zero's at masked positions
        return self.conv(x)
    

class VerticalStackConvolution(MaskedConvolution):
    
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0
        
        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size//2,:] = 0
        
        super().__init__(c_in, c_out, mask, **kwargs)
        
class HorizontalStackConvolution(MaskedConvolution):
    
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0
        
        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0,kernel_size//2] = 0
        
        super().__init__(c_in, c_out, mask, **kwargs)

class GatedMaskedConv(nn.Module):
    
    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)
    
    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)
        
        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack
        
        return v_stack_out, h_stack_out
    

def create_checkpoint_dir():

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, "PixelCNN")):
        os.makedirs(os.path.join(models_dir, "PixelCNN"))


class PixelCNN(nn.Module):
    
    def __init__(self, c_in, c_hidden):
        super().__init__()
        
        self.channels = c_in
        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=4),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden)
        ])
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, c_in * 256, kernel_size=1, padding=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with float values between -1 and 1.
        """
        x = x/255.0 * 2 - 1
        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        out = self.conv_out(F.elu(h_stack))
        
        # Output dimensions: [Batch, Classes, Channels, Height, Width]
        out = out.reshape(out.shape[0], 256, out.shape[1]//256, out.shape[2], out.shape[3])
        return out
    
    def calc_likelihood(self, x, train=True):
        # Forward pass with bpd likelihood calculation
        if train:
            pred = self.forward(x)
            nll = F.cross_entropy(pred, x, reduction='none')
            bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))
            return bpd.mean()
        else:
            with torch.no_grad():
                pred = self.forward(x)
                nll = F.cross_entropy(pred, x, reduction='none')
                bpd = nll.mean(dim=[1,2,3]) * np.log2(np.exp(1))
            return bpd
        
    @torch.no_grad()
    def sample(self, img_shape, img=None, train=False):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill 
                             should be -1 in the input tensor.
        """
        # Create empty image
        if img is None:
            img = torch.zeros(img_shape, dtype=torch.float).to(self.device) - 1
        # Generation loop
        for h in tqdm(range(img_shape[2]), leave=False, desc="Sampling"):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # Skip if not to be filled (-1)
                    if (img[:,c,h,w] != -1).all().item():
                        continue
                    # For efficiency, we only have to input the upper part of the image
                    # as all other parts will be skipped by the masked convolutions anyways
                    pred = self.forward(img[:,:,:h+1,:]) 
                    probs = F.softmax(pred[:,:,c,h,w], dim=-1)
                    img[:,c,h,w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        
        # print min and max values
        # clip to 0-255 and convert to uint8
        fig = plt.figure(figsize=(10,10))
        img = (img*255.0).clip(0,255).to(torch.uint8)
        grid = make_grid(img, nrow=np.sqrt(img_shape[0]).astype(int))
        grid = grid.permute(1,2,0).cpu().numpy()
        plt.imshow(grid)
        plt.axis('off')

        if train:
            wandb.log({"Samples": fig})
        else:
            plt.show()
    
    def configure_optimizers(self, args):
        optimizer = torch.optim.Adam(self.parameters(), lr = args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.gamma)
        return optimizer, scheduler
    
    def training_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch)                             
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch)
    
    def test_step(self, batch, batch_idx):
        loss = self.calc_likelihood(batch)

    def train_model(self, dataloader, args, img_size=32):

        create_checkpoint_dir()
        optimizer, scheduler = self.configure_optimizers(args)
        epoch_bar = tqdm(range(args.n_epochs), desc="Epochs")
        best_loss = np.inf

        for epoch in epoch_bar:
            
            self.train()
            loss_acc = 0.0
            for batch,_ in tqdm(dataloader, desc="Batches", leave=False):

                batch = (batch*255.0).clip(0,255).to(torch.long)
                batch = batch.to(self.device)

                optimizer.zero_grad()
                loss = self.training_step(batch, 0)

                loss.backward()
                optimizer.step()
                loss_acc += loss.item()*batch.shape[0]
                wandb.log({"BPD Loss": loss.item()})

            scheduler.step()
            epoch_bar.set_postfix({"Loss": loss_acc/len(dataloader.dataset)})

            if (epoch+1) % args.sample_and_save_freq == 0 or epoch == 0:
                self.eval()
                self.sample((16,self.channels,img_size,img_size), train=True)
            
            if best_loss > loss_acc:
                best_loss = loss_acc
                torch.save(self.state_dict(), os.path.join(models_dir, "PixelCNN", f"PixelCNN_{args.dataset}.pt"))

    def outlier_detection(self, in_loader, out_loader):

        self.eval()
        in_scores = []
        out_scores = []

        for batch,_ in tqdm(in_loader, desc="Inlier Detection"):
            batch = (batch*255.0).clip(0,255).to(torch.long)
            batch = batch.to(self.device)
            loss = self.calc_likelihood(batch, train=False)
            in_scores.append(loss.cpu().numpy())
        
        in_scores = np.concatenate(in_scores)

        for batch,_ in tqdm(out_loader, desc="Outlier Detection"):
            batch = (batch*255.0).clip(0,255).to(torch.long)
            batch = batch.to(self.device)
            loss = self.calc_likelihood(batch, train=False)
            out_scores.append(loss.cpu().numpy())
        
        out_scores = np.concatenate(out_scores)

        # Plot histograms in a single plot, no subplots
        fig = plt.figure(figsize=(10,5))
        plt.hist(in_scores, bins=50, alpha=0.5, label="Inlier")
        plt.hist(out_scores, bins=50, alpha=0.5, label="Outlier")
        plt.legend()
        plt.xlabel("BPD")
        plt.ylabel("Count")
        plt.title("PixelCNN Outlier Detection")
        plt.show()

        return in_scores, out_scores
