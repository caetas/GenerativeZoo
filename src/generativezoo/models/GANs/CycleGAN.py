import torch.nn as nn
import torch.nn.functional as F
import itertools
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import wandb
from config import models_dir

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def train(dataloader_A, dataloader_B, testloader_A, testloader_B, n_epochs, lr, decay, in_channels, out_channels, device, sample_and_save_freq=5, name='horse2zebra'):

    G_AB = Generator(in_channels, out_channels).to(device)
    G_BA = Generator(out_channels, in_channels).to(device)
    D_A = Discriminator(in_channels).to(device)
    D_B = Discriminator(out_channels).to(device)

    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, 0, decay).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, 0, decay).step)

    step_per_epoch = min(len(dataloader_A), len(dataloader_B))

    best_loss = np.inf

    for epoch in tqdm(range(n_epochs), desc='Epochs'):

        acc_loss_G = 0
        acc_loss_G_GAN = 0
        acc_loss_G_cycle = 0
        acc_loss_G_identity = 0
        acc_loss_D_A = 0
        acc_loss_D_B = 0
        elements = 0

        for s in tqdm(range(step_per_epoch), desc='Steps', leave=False):
                
            # Set model input
            real_A = next(iter(dataloader_A)).to(device)
            real_B = next(iter(dataloader_B)).to(device)

            # Adversarial ground truths
            valid = torch.ones(real_A.size(0), 1).to(device)
            fake = torch.zeros(real_A.size(0), 1).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)

            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)

            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            acc_loss_D_A += loss_D_A.item() * real_A.size(0)
            acc_loss_D_B += loss_D_B.item() * real_B.size(0)
            acc_loss_G += loss_G.item() * real_A.size(0)
            acc_loss_G_GAN += loss_GAN.item() * real_A.size(0)
            acc_loss_G_cycle += loss_cycle.item() * real_A.size(0)
            acc_loss_G_identity += loss_identity.item() * real_A.size(0)
            elements += real_A.size(0)

            # --------------
        #  Log Progress
        # --------------
        wandb.log({'loss_G': acc_loss_G/elements, 'loss_G_GAN': acc_loss_G_GAN/elements, 'loss_G_cycle': acc_loss_G_cycle/elements, 'loss_G_identity': acc_loss_G_identity/elements, 'loss_D_A': acc_loss_D_A/elements, 'loss_D_B': acc_loss_D_B/elements, 'epoch': epoch})
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        if acc_loss_G/elements < best_loss:
            best_loss = acc_loss_G/elements
            torch.save(G_AB.state_dict(), models_dir + '/CycleGAN_{}_G_AB.pth'.format(name))
            torch.save(G_BA.state_dict(), models_dir + '/CycleGAN_{}_G_BA.pth'.format(name))

        if epoch % sample_and_save_freq == 0:

            # select a batch of real samples
            real_A = next(iter(testloader_A)).to(device)
            real_B = next(iter(testloader_B)).to(device)

            with torch.no_grad():
                # generate a batch of fake samples
                fake_A = G_BA(real_B)
                fake_B = G_AB(real_A)

            real_A = real_A.detach().cpu().numpy()
            real_B = real_B.detach().cpu().numpy()
            fake_A = fake_A.detach().cpu().numpy()
            fake_B = fake_B.detach().cpu().numpy()

            real_A = np.transpose(real_A, (0, 2, 3, 1))*0.5 + 0.5
            real_B = np.transpose(real_B, (0, 2, 3, 1))*0.5 + 0.5
            fake_A = np.transpose(fake_A, (0, 2, 3, 1))*0.5 + 0.5
            fake_B = np.transpose(fake_B, (0, 2, 3, 1))*0.5 + 0.5

            # plot an image with real_A.shape[0] rows and 4 columns (real_A, fake_B, real_B, fake_A)
            n_rows = real_A.shape[0]
            n_cols = 4
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5 * (n_rows // n_cols)))
            for i, ax in enumerate(axes.flat):
                if i % 4 == 0:
                    ax.imshow(real_A[i//4])
                elif i % 4 == 1:
                    ax.imshow(fake_B[i//4])
                elif i % 4 == 2:
                    ax.imshow(real_B[i//4])
                elif i % 4 == 3:
                    ax.imshow(fake_A[i//4])
                ax.axis('off')

            wandb.log({'samples': wandb.Image(fig)})
            plt.close(fig)

def test(checkpoint_A, checkpoint_B, test_dataloader_A, test_dataloader_B, in_channels, out_channels, device):

    G_AB = Generator(in_channels, out_channels).to(device)
    G_BA = Generator(out_channels, in_channels).to(device)

    G_AB.load_state_dict(torch.load(checkpoint_A))
    G_BA.load_state_dict(torch.load(checkpoint_B))

    # select a batch of real samples
    real_A = next(iter(test_dataloader_A)).to(device)
    real_B = next(iter(test_dataloader_B)).to(device)

    with torch.no_grad():
        # generate a batch of fake samples
        fake_A = G_BA(real_B)
        fake_B = G_AB(real_A)

    real_A = real_A.detach().cpu().numpy()
    real_B = real_B.detach().cpu().numpy()
    fake_A = fake_A.detach().cpu().numpy()
    fake_B = fake_B.detach().cpu().numpy()

    real_A = np.transpose(real_A, (0, 2, 3, 1))*0.5 + 0.5
    real_B = np.transpose(real_B, (0, 2, 3, 1))*0.5 + 0.5
    fake_A = np.transpose(fake_A, (0, 2, 3, 1))*0.5 + 0.5
    fake_B = np.transpose(fake_B, (0, 2, 3, 1))*0.5 + 0.5

    # plot an image with real_A.shape[0] rows and 4 columns (real_A, fake_B, real_B, fake_A)
    n_rows = real_A.shape[0]
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i % 4 == 0:
            ax.imshow(real_A[i//4])
        elif i % 4 == 1:
            ax.imshow(fake_B[i//4])
        elif i % 4 == 2:
            ax.imshow(real_B[i//4])
        elif i % 4 == 3:
            ax.imshow(fake_A[i//4])
        ax.axis('off')

    plt.show()