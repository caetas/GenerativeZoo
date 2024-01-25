import torch
from torchvision.transforms import Compose, Lambda, ToPILImage
from matplotlib import pyplot as plt
import numpy as np
from data.Dataloaders import *
from models.Diffusion.Diffusion import *
from utils.util import parse_args_DDPM
from config import data_raw_dir, models_dir
import mlflow

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args_DDPM()
normalize = True

def plot_samples(samples):
     n_rows = int(np.sqrt(samples.shape[0]))
     n_cols = n_rows
     samples = np.transpose(samples, (0, 2, 3, 1))
     fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
     for i, ax in enumerate(axes.flat):
          ax.imshow(samples[i].squeeze(), cmap='gray')
          ax.axis('off')
     plt.show()


reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

if args.train:
     mlflow.set_experiment('DDPM')
     mlflow.start_run()
     mlflow.set_tag('mlflow.runName', 'DDPM_' + args.dataset)
     mlflow.log_param('dataset', args.dataset)
     mlflow.log_param('batch_size', args.batch_size)
     mlflow.log_param('n_epochs', args.n_epochs)
     mlflow.log_param('lr', args.lr)
     mlflow.log_param('timesteps', args.timesteps)
     mlflow.log_param('beta_start', args.beta_start)
     mlflow.log_param('beta_end', args.beta_end)
     mlflow.log_param('ddpm', args.ddpm)
     scheduler = LinearScheduler(beta_start=args.beta_start, beta_end=args.beta_end, timesteps=args.timesteps)
     forward_diffusion = ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, reverse_transform=reverse_transform)
     dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize)
     mlflow.log_param('input_size', input_size)
     mlflow.log_param('channels', channels)
     sampler = Sampler(betas=scheduler.betas, timesteps=args.timesteps, reduced_timesteps=args.timesteps, ddpm=args.ddpm)
     model = DDPM(n_features=input_size, in_channels=channels, channel_scale_factors=(1, 2, 4,)).to(device)
     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
     criterion = get_loss
     train(image_size=input_size, num_channels=channels, dataloader=dataloader, device=device, epochs=args.n_epochs, timesteps=args.timesteps, sample_and_save_freq=args.sample_and_save_freq, forward_diffusion_model=forward_diffusion, denoising_model=model, criterion=criterion, optimizer=optimizer, sampler=sampler, loss_type=args.loss_type)
     mlflow.end_run()

elif args.sample:
     scheduler = LinearScheduler(beta_start=args.beta_start, beta_end=args.beta_end, timesteps=args.timesteps)
     sampler = Sampler(betas=scheduler.betas, timesteps=args.timesteps, reduced_timesteps=args.sample_timesteps, ddpm=args.ddpm)
     _, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize)
     model = DDPM(n_features=input_size, in_channels=channels, channel_scale_factors=(1, 2, 4,)).to(device)
     model.load_state_dict(torch.load(args.checkpoint))
     samps = sampler.sample(model=model, image_size=input_size, batch_size=args.num_samples, channels=channels)[-1]
     plot_samples(samps)

elif args.outlier_detection:
     scheduler = LinearScheduler(beta_start=args.beta_start, beta_end=args.beta_end, timesteps=args.timesteps)
     forward_diffusion_model = ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, reverse_transform=reverse_transform)
     dataloader_a, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize)
     model = DDPM(n_features=input_size, in_channels=channels, channel_scale_factors=(1, 2, 4,)).to(device)
     model.load_state_dict(torch.load(args.checkpoint))
     dataloader_b, input_size_b, channels_b = pick_dataset(args.out_dataset, 'val', args.batch_size, normalize=normalize)
     outlier_detection(denoising_model=model, val_loader=dataloader_a, out_loader=dataloader_b, device=device, forward_diffusion_model=forward_diffusion_model, loss_type=args.loss_type)
     pass
else:
     raise ValueError('Please specify at least one of the following: train, sample, outlier_detection')