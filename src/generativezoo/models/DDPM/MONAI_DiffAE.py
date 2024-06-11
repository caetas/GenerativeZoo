##################################################################################################################################################################
### Based on https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_diffusion_autoencoder/2d_diffusion_autoencoder_tutorial.ipynb ###
##################################################################################################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm, trange
from sklearn.linear_model import LogisticRegression
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
from config import models_dir
import wandb

def create_checkpoint_dir():
  if not os.path.exists(models_dir):
    os.makedirs(models_dir)
  if not os.path.exists(os.path.join(models_dir, 'DiffusionAE')):
    os.makedirs(os.path.join(models_dir, 'DiffusionAE'))

class DiffAE(nn.Module):
    def __init__(self, args, in_channels = 3):
        '''Diffusion Autoencoder model
        Args:
            args (argparse.ArgumentParser): the arguments to configure the model
            in_channels (int): the number of input channels
        '''
        super(DiffAE, self).__init__()
        self.embedding_dimension = args.embedding_dimension
        self.encoder = torchvision.models.resnet18()
        self.encoder.fc = nn.Linear(512, self.embedding_dimension)
        if in_channels == 1:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.unet = DiffusionModelUNet(
                    spatial_dims=2,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    num_channels=args.model_channels,
                    attention_levels=args.attention_levels,
                    num_res_blocks=args.num_res_blocks,
                    num_head_channels=64,
                    with_conditioning=True,
                    cross_attention_dim=1,
                )
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.torch_device)
        self.unet.to(self.torch_device)
        self.scheduler = DDIMScheduler(num_train_timesteps = args.num_train_timesteps)
        self.inferer = DiffusionInferer(self.scheduler)
        self.inference_timesteps = args.inference_timesteps
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.sample_and_save_freq = args.sample_and_save_freq
        self.dataset = args.dataset

    def forward(self, xt, x_cond, t):
        '''Forward pass of the model
        Args:
            xt (torch.Tensor): the input image
            x_cond (torch.Tensor): the conditioning image
            t (torch.Tensor.long): the timestep
        Returns:
            pred (torch.Tensor): the predicted noise
            latent (torch.Tensor): the latent representation of the input image
            '''
        latent = self.encoder(x_cond)
        pred = self.unet(x=xt, context = latent, timesteps = t)
        return pred, latent
    
    def generate_samples(self, val_loader, name = "generic", train = False):
        '''Generate samples from the model
        Args:
            val_loader (torch.utils.data.DataLoader): the validation data loader
            name (str): the name of the file to save the samples
            train (bool): whether we are training the model or just sampling
        '''
        self.eval()
        self.scheduler.set_timesteps(num_inference_steps=self.inference_timesteps)
        batch = next(iter(val_loader))
        images = batch[0].to(self.torch_device)        
        noise = torch.randn_like(images).to(self.torch_device)
        latent = self.encoder(images)
        reconstruction = self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler, save_intermediates=False, conditioning=latent.unsqueeze(2))

        images = images*0.5 + 0.5
        reconstruction = reconstruction*0.5 + 0.5

        grid = torchvision.utils.make_grid(torch.cat([images[:8],reconstruction[:8]]), nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0)
        fig = plt.figure(figsize=(15,5))
        plt.imshow(grid.detach().cpu().numpy().transpose(1,2,0))
        plt.axis('off')
        if train:
            wandb.log({"Samples": fig})
        else:
            plt.show()
        plt.close(fig)

    def evaluate(self, val_loader):
        '''Evaluate the model on the validation set
        Args:
            val_loader (torch.utils.data.DataLoader): the validation data loader
        Returns:
            val_loss (float): the validation loss
        '''
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(self.torch_device)
                noise = torch.randn_like(images).to(self.torch_device)
                timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.size(0),)).to(self.torch_device).long()
                latent = self.encoder(images)
                noise_pred = self.inferer(inputs=images, diffusion_model=self.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
                loss = F.mse_loss(noise_pred.float(), noise.float())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        return val_loss
    
    def train_model(self, train_loader, val_loader):
        '''Train the model
        Args:
            train_loader (torch.utils.data.DataLoader): the training data loader
            val_loader (torch.utils.data.DataLoader): the validation data loader
            name (str): the name of the file to save the samples
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        best_loss = np.inf
        create_checkpoint_dir()

        epoch_bar = trange(self.n_epochs)
        for epoch in epoch_bar:
            self.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, leave=False, desc = f"Epoch {epoch+1}/{self.n_epochs}"):
                images = batch[0].to(self.torch_device)
                noise = torch.randn_like(images).to(self.torch_device)
                timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.size(0),)).to(self.torch_device).long()
                optimizer.zero_grad()
                latent = self.encoder(images)
                noise_pred = self.inferer(inputs=images, diffusion_model=self.unet, noise=noise, timesteps=timesteps, condition = latent.unsqueeze(2))
                loss = F.mse_loss(noise_pred.float(), noise.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            epoch_bar.set_description(f"Train Loss: {train_loss}")
            wandb.log({"Train Loss": train_loss}, step = epoch)

            if val_loader and ((epoch + 1) % self.sample_and_save_freq == 0 or epoch == 0):
                val_loss = self.evaluate(val_loader)
                epoch_bar.set_description(f"Train Loss: {train_loss} - Val Loss: {val_loss}")
                self.generate_samples(val_loader, name = f"epoch_{str(epoch)}", train = True)
                wandb.log({"Val Loss": val_loss}, step = epoch)

            # save model if it has the best val loss
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(self.state_dict(), os.path.join(models_dir,'DiffusionAE', f"DiffAE_{self.dataset}.pt"))

    def linear_regression(self, train_loader, val_loader):
        '''Evaluate the latent space of the model
        Args:
            train_loader (torch.utils.data.DataLoader): the training data loader
            val_loader (torch.utils.data.DataLoader): the validation data loader
        '''
        latent_train = []
        labels_train = []
        latent_val = []
        labels_val = []

        self.encoder.eval()
        with torch.no_grad():
            for batch in train_loader:
                images, labels = batch
                images = images.to(self.torch_device)
                latent = self.encoder(images)
                latent_train.append(latent.cpu().numpy())
                labels_train.append(labels.numpy())
            for batch in val_loader:
                images, labels = batch
                images = images.to(self.torch_device)
                latent = self.encoder(images)
                latent_val.append(latent.cpu().numpy())
                labels_val.append(labels.numpy())
        latent_train = np.concatenate(latent_train, axis=0)
        labels_train = np.concatenate(labels_train, axis=0).squeeze()
        latent_val = np.concatenate(latent_val, axis=0)
        labels_val = np.concatenate(labels_val, axis=0).squeeze()
        clf = LogisticRegression(solver = 'newton-cg', random_state=0).fit(latent_train, labels_train)
        
        self.w = torch.tensor(clf.coef_).float().to(self.torch_device)

        train_acc = clf.score(latent_train, labels_train)
        val_acc = clf.score(latent_val, labels_val)

        self.clf = clf

        print(f"Train Accuracy: {train_acc} - Val Accuracy: {val_acc}")

    def test_model(self, test_loader):
        '''Test the model
        Args:
            test_loader (torch.utils.data.DataLoader): the test data loader
        '''
        self.eval()
        predicted = []
        true = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(self.torch_device)
                labels = batch[1]
                latent = self.encoder(images)
                predicted_labels = self.clf.predict(latent.cpu().numpy())
                predicted.append(predicted_labels)
                true.append(labels.numpy().squeeze())
        predicted = np.concatenate(predicted, axis=0)
        true = np.concatenate(true, axis=0)
        # count correct predictions
        accuracy = np.sum(predicted == true)/len(true)
        # f1 score
        precision = np.sum((predicted == true) & (predicted == 1))/np.sum(predicted == 1)
        recall = np.sum((predicted == true) & (predicted == 1))/np.sum(true == 1)
        f1 = 2*precision*recall/(precision + recall)
        print(f"Test F1 Score: {f1}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test Accuracy: {accuracy}")

    def manipulate_latent(self, val_loader, name = "manipulated"):
        '''Manipulate the latent space of the model
        Args:
            val_loader (torch.utils.data.DataLoader): the validation data loader
            name (str): the name of the file to save the samples
        '''
        self.eval()
        batch = next(iter(val_loader))
        images = batch[0].to(self.torch_device)
        labels = batch[1]
        latent = self.encoder(images)
        latent_manipulated = latent - 1.5*self.w
        # predict new labels
        new_labels = self.clf.predict(latent_manipulated.detach().cpu().numpy())
        print(f"New Labels: {new_labels[:8]}")
        print(f"Original Labels: {labels[:8].cpu().numpy().squeeze()}")
        self.scheduler.set_timesteps(num_inference_steps=self.inference_timesteps)
        noise = torch.randn_like(images).to(self.torch_device)
        reconstruction = self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler, save_intermediates=False, conditioning=latent.unsqueeze(2))
        reconstruction_manipulated = self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler, save_intermediates=False, conditioning=latent_manipulated.unsqueeze(2))

        images = images*0.5 + 0.5
        reconstruction = reconstruction*0.5 + 0.5
        reconstruction_manipulated = reconstruction_manipulated*0.5 + 0.5


        grid = torchvision.utils.make_grid(torch.cat([images[:8],reconstruction[:8], reconstruction_manipulated[:8]]), nrow=8, padding=2, normalize=False, scale_each=False, pad_value=0)
        plt.figure(figsize=(15,5))
        plt.imshow(grid.detach().cpu().numpy().transpose(1,2,0))
        plt.axis('off')
        plt.show()
    
    def manipulate_image(self, image, transformation):
        '''Manipulate an image
        Args:
            image (torch.Tensor): the input image
            transformation (float): the transformation to apply to the image
        Returns:
            manipulated_image (torch.Tensor): the manipulated image
        '''
        self.eval()
        image = image.unsqueeze(0).to(self.torch_device)
        latent = self.encoder(image)
        latent_manipulated = latent + transformation*self.w
        self.scheduler.set_timesteps(num_inference_steps=self.inference_timesteps)
        noise = torch.randn_like(image).to(self.torch_device)
        reconstruction_manipulated = self.inferer.sample(input_noise=noise, diffusion_model=self.unet, scheduler=self.scheduler, save_intermediates=False, conditioning=latent_manipulated.unsqueeze(2))
        manipulated_image = reconstruction_manipulated*0.5 + 0.5
        manipulated_image = manipulated_image.squeeze(0).cpu().detach().numpy()
        manipulated_image = np.transpose(manipulated_image, (1,2,0))
        # predict new label
        new_label = self.clf.predict(latent_manipulated.detach().cpu().numpy())
        return manipulated_image, new_label
