##############################################################################################
################# Code based on: https://github.com/adjidieng/PresGANs/ ######################
##############################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm, trange
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from config import models_dir
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
 
class Generator(nn.Module):
    def __init__(self, imgSize, nz, ngf, nc):
        super(Generator, self).__init__()
        if imgSize < 64:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 4 x 4
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 8 x 8
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 16 x 16
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 32 x 32
            )
        elif imgSize == 64:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,    ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        
        elif imgSize == 128:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*16) x 4 x 4
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 16 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 32 x 32
                nn.ConvTranspose2d(ngf * 2,    ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 64 x 64
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 128 x 128
            )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, imgSize, ndf, nc):
        super(Discriminator, self).__init__()
        
        if imgSize < 64:
            self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 16 x 16
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 8 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        elif imgSize == 64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        
        elif imgSize == 128:
            self.main = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 64 x 64
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 32 x 32
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 16 x 16
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 8 x 8
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*16) x 4 x 4
                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
    
def _helper(netG, x_tilde, eps, sigma):
    eps = eps.clone().detach().requires_grad_(True)
    with torch.no_grad():
        G_eps = netG(eps)
    bsz = eps.size(0)
    log_prob_eps = (eps ** 2).view(bsz, -1).sum(1).view(-1, 1)
    log_prob_x = (x_tilde - G_eps)**2 / sigma**2
    log_prob_x = log_prob_x.view(bsz, -1)
    log_prob_x = torch.sum(log_prob_x, dim=1).view(-1, 1)
    logjoint_vect = -0.5 * (log_prob_eps + log_prob_x)
    logjoint_vect = logjoint_vect.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = eps.grad
    return logjoint_vect, logjoint, grad_logjoint

def get_samples(netG, x_tilde, eps_init, sigma, burn_in, num_samples_posterior, 
            leapfrog_steps, stepsize, flag_adapt, hmc_learning_rate, hmc_opt_accept):
    device = eps_init.device
    bsz, eps_dim = eps_init.size(0), eps_init.size(1)
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz*num_samples_posterior, eps_dim).to(device)
    current_eps = eps_init
    cnt = 0
    for i in range(n_steps):
        eps = current_eps
        p = torch.randn_like(current_eps)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, current_eps, sigma)
        current_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            eps = eps + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, eps, sigma)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, eps, sigma)  
        proposed_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p**2).sum(dim=1) 
        current_K = current_K.view(-1, 1) ## should be size of B x 1 
        proposed_K = 0.5 * (p**2).sum(dim=1) 
        proposed_K = proposed_K.view(-1, 1) ## should be size of B x 1 
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K)) 
        accept = accept.float().squeeze() ## should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try: 
            len(ind) > 0
            current_eps[ind, :] = eps[ind, :]  
            current_U[ind] = proposed_U[ind]
        except:
            print('Samples were all rejected...skipping')
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        else:
            samples[cnt*bsz : (cnt+1)*bsz, :] = current_eps.squeeze()
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'PrescribedGAN')):
    os.makedirs(os.path.join(models_dir, 'PrescribedGAN'))

class PresGAN(nn.Module):
    def __init__(self, imgSize, nz, ngf, ndf, nc, device, beta1, lrD, lrG, sigma_lr, n_epochs, num_gen_images, restrict_sigma, sigma_min, sigma_max, stepsize_num, lambda_, burn_in, num_samples_posterior, leapfrog_steps, flag_adapt, hmc_learning_rate, hmc_opt_accept, dataset='cifar10', sample_and_save_freq=5):
        super(PresGAN, self).__init__()
        self.netG = Generator(imgSize, nz, ngf, nc).to(device)
        self.netD = Discriminator(imgSize, ndf, nc).to(device)
        self.log_sigma = nn.Parameter(torch.zeros(1, imgSize, imgSize, requires_grad=True, device=device))
        self.imgSize = imgSize
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.beta1 = beta1
        self.lrD = lrD
        self.lrG = lrG
        self.sigma_lr = sigma_lr
        self.n_epochs = n_epochs
        self.num_gen_images = num_gen_images
        self.device = device
        self.restrict_sigma = restrict_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.stepsize_num = stepsize_num
        self.lambda_ = lambda_
        self.burn_in = burn_in
        self.num_samples_posterior = num_samples_posterior
        self.leapfrog_steps = leapfrog_steps
        self.flag_adapt = flag_adapt
        self.hmc_learning_rate = hmc_learning_rate
        self.hmc_opt_accept = hmc_opt_accept
        self.dataset = dataset
        self.sample_and_save_freq = sample_and_save_freq
    
    def forward(self, input):
        return self.netG(input)
    
    def train_model(self, dataloader):
        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()
        criterion_mse = nn.MSELoss()

        fixed_noise = torch.randn(self.num_gen_images, self.nz, 1, 1, device=self.device)
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(self.beta1, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(self.beta1, 0.999)) 
        sigma_optimizer = torch.optim.Adam([self.log_sigma], lr=self.sigma_lr, betas=(self.beta1, 0.999))

        if self.restrict_sigma:
            logsigma_min = math.log(math.exp(self.sigma_min) - 1.0)
            logsigma_max = math.log(math.exp(self.sigma_max) - 1.0)
        stepsize = self.stepsize_num / self.nz

        epoch_bar = trange(self.n_epochs, desc = "Loss: ----", leave = True)
        best_loss = np.inf

        create_checkpoint_dir()

        for epoch in epoch_bar:
            for x,_ in tqdm(dataloader, desc='Batches', leave=False):
                x = x.to(self.device)
                sigma_x = F.softplus(self.log_sigma).view(1, 1, self.imgSize, self.imgSize)

                self.netD.zero_grad()
                
                label = torch.full((x.shape[0],), real_label, device=self.device, dtype=torch.float32)

                noise_eta = torch.randn_like(x)
                noised_data = x + sigma_x.detach() * noise_eta
                out_real = self.netD(noised_data)
                errD_real = criterion(out_real, label)
                errD_real.backward()
                D_x = out_real.mean().item()

                # train with fake
                
                noise = torch.randn(x.shape[0], self.nz, 1, 1, device=self.device)
                mu_fake = self.netG(noise) 
                fake = mu_fake + sigma_x * noise_eta
                label.fill_(fake_label)
                out_fake = self.netD(fake.detach())
                errD_fake = criterion(out_fake, label)
                errD_fake.backward()
                D_G_z1 = out_fake.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                # update G network: maximize log(D(G(z)))

                self.netG.zero_grad()
                sigma_optimizer.zero_grad()

                label.fill_(real_label)  
                gen_input = torch.randn(x.shape[0], self.nz, 1, 1, device=self.device)
                out = self.netG(gen_input)
                noise_eta = torch.randn_like(out)
                g_fake_data = out + noise_eta * sigma_x

                dg_fake_decision = self.netD(g_fake_data)
                g_error_gan = criterion(dg_fake_decision, label) 
                D_G_z2 = dg_fake_decision.mean().item()

                if self.lambda_ == 0:
                    g_error_gan.backward()
                    optimizerG.step() 
                    sigma_optimizer.step()
                else:
                    hmc_samples, acceptRate, stepsize = get_samples(self.netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), self.burn_in, self.num_samples_posterior, self.leapfrog_steps, stepsize, self.flag_adapt, self.hmc_learning_rate, self.hmc_opt_accept)

                    bsz, d = hmc_samples.size()
                    mean_output = self.netG(hmc_samples.view(bsz, d, 1, 1).to(self.device))
                    bsz = g_fake_data.size(0)

                    mean_output_summed = torch.zeros_like(g_fake_data)
                    for cnt in range(self.num_samples_posterior):
                        mean_output_summed = mean_output_summed + mean_output[cnt*bsz:(cnt+1)*bsz]
                    mean_output_summed = mean_output_summed / self.num_samples_posterior  

                    c = ((g_fake_data - mean_output_summed) / sigma_x**2).detach()
                    g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                    g_error = g_error_gan - self.lambda_ * g_error_entropy
                    g_error.backward()
                    optimizerG.step() 
                    sigma_optimizer.step()

                if self.restrict_sigma:
                    self.log_sigma.data.clamp_(min=logsigma_min, max=logsigma_max)
            
            epoch_bar.set_description("Loss_D: {:.4f}, Loss_G: {:.4f}".format(errD.item(), g_error_gan.item()))
            epoch_bar.refresh()
            # Log the losses
            wandb.log({"Loss_D": errD.item(), "Loss_G": g_error.item(), "Loss_G_GAN": g_error_gan.item(), "Loss_G_Entropy": g_error_entropy.item()})

            if g_error_gan.item() < best_loss:
                best_loss = g_error_gan.item()
                torch.save(self.netG.state_dict(), os.path.join(models_dir, 'PrescribedGAN', f"PresGAN_{self.dataset}_{self.nz}.pt"))
                torch.save(self.netD.state_dict(), os.path.join(models_dir, 'PrescribedGAN', f"PresDisc_{self.dataset}_{self.nz}.pt"))
                torch.save(self.log_sigma, os.path.join(models_dir, 'PrescribedGAN', f"PresSigma_{self.dataset}_{self.nz}.pt"))

            if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:
                with torch.no_grad():
                    fake = self.netG(fixed_noise).detach()
                    noise_eta = torch.randn_like(fake)
                    #fake = fake + noise_eta * sigma_x
                    fake = fake.cpu()
                    fake = fake.clamp(-1, 1)
                    fake = fake*0.5 + 0.5
                    nrow = int(np.sqrt(self.num_gen_images))
                    img_grid = make_grid(fake, nrow=nrow, padding=2)
                    fig = plt.figure(figsize=(10,10))
                    plt.imshow(np.transpose(img_grid, (1,2,0)))
                    plt.axis('off')
                    #plt.savefig(f"PresGAN_{self.dataset}_epoch_{epoch}.png")
                    wandb.log({"Samples": fig})
                    plt.close(fig)

    def load_checkpoints(self,generator_checkpoint=None, discriminator_checkpoint=None, sigma_checkpoint=None):
        if generator_checkpoint is not None:
            self.netG.load_state_dict(torch.load(generator_checkpoint))
        if discriminator_checkpoint is not None:
            self.netD.load_state_dict(torch.load(discriminator_checkpoint))
        if sigma_checkpoint is not None:
            self.log_sigma = torch.load(sigma_checkpoint)
    
    @torch.no_grad()
    def sample(self, num_samples=16):
        fixed_noise = torch.randn(num_samples, self.nz, 1, 1, device=self.device)
        fake = self.netG(fixed_noise).detach().cpu()
        fake = fake*0.5 + 0.5
        nrow = int(np.sqrt(num_samples))
        img_grid = make_grid(fake, nrow=nrow, padding=2)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(np.transpose(img_grid, (1,2,0)))
        plt.axis('off')
        plt.show()
        plt.close(fig)

    @torch.no_grad()
    def outlier_detection(self, in_loader, out_loader, in_array=None, display=True):
        #just get the discriminator scores
        if in_array is not None:
            in_scores = in_array
        else:
            in_scores = []
            for x,_ in tqdm(in_loader, desc='In-distribution', leave=False):
                x = x.to(self.device)
                sigma_x = F.softplus(self.log_sigma).view(1, 1, self.imgSize, self.imgSize)
                noise_eta = torch.randn_like(x)
                noised_data = x + sigma_x.detach() * noise_eta
                out_real = self.netD(noised_data)
                in_scores.append(out_real.cpu().numpy())
            in_scores = np.concatenate(in_scores)
            in_scores = -in_scores + 1

        out_scores = []
        for x,_ in tqdm(out_loader, desc='Out-of-distribution', leave=False):
            x = x.to(self.device)
            sigma_x = F.softplus(self.log_sigma).view(1, 1, self.imgSize, self.imgSize)
            noise_eta = torch.randn_like(x)
            noised_data = x + sigma_x.detach() * noise_eta
            out_real = self.netD(noised_data)
            out_scores.append(out_real.cpu().numpy())
        out_scores = np.concatenate(out_scores)
        out_scores = -out_scores + 1

        # calculate AUROC
        in_labels = np.zeros_like(in_scores)
        out_labels = np.ones_like(out_scores)
        labels = np.concatenate([in_labels, out_labels])
        scores = np.concatenate([in_scores, out_scores])
        auroc = roc_auc_score(labels, scores)

        # get the ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        # get fpr at tpr=0.95
        fpr95 = fpr[np.argmax(tpr >= 0.95)]

        if display:
            print(f'AUROC: {auroc:.6f}, FPR95: {fpr95:.6f}, Mean Scores: {np.mean(out_scores):.6f}')
            # plot histograms
            plt.hist(in_scores, bins=50, alpha=0.5, label='In-distribution')
            plt.hist(out_scores, bins=50, alpha=0.5, label='Out-of-distribution')
            plt.title(f'AUROC: {auroc:.4f}')
            plt.legend(loc='upper left')
            plt.show()

        return auroc, fpr95, in_scores, np.mean(out_scores)
    