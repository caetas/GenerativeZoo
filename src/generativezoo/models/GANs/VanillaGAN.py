from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

##########################################################################################
### https://github.com/TeeyoHuang/conditional-GAN/blob/master/conditional_DCGAN.py     ###
##########################################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dim, d=128, channels=3):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, d*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, 4, 2, 1)


    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(channels, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)


    # def forward(self, input):
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x

def train(dataloader, n_epochs, device, lr = 0.0002, beta1 = 0.5, beta2 = 0.999, latent_dim = 100, n_classes = 10, img_size = 32, channels = 3, sample_interval = 5):

    # Initialize generator and discriminator
    generator = Generator(latent_dim = latent_dim, channels=channels).to(device)
    discriminator = Discriminator(channels=channels).to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    for epoch in tqdm(range(n_epochs)):
        for (imgs, _) in tqdm(dataloader, leave=False):

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            # Configure input
            real_imgs = imgs.to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs).view(-1, 1)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs).view(-1, 1)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach()).view(-1, 1)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        if epoch % sample_interval == 0:
            # create row of n_classes images
            z = torch.randn(16, latent_dim, 1, 1).to(device)
            gen_imgs = generator(z)
            # plot images
            plt.figure(figsize=(10, 10))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.imshow(gen_imgs[i].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5, cmap='gray')
                plt.axis('off')
            plt.show()