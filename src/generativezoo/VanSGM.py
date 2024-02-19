from models.SGM.VanillaSGM import *
from data.Dataloaders import *
from utils.util import parse_args_VanillaSGM
from config import models_dir
import torch
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args_VanillaSGM()
normalize = True

if args.train:
    wandb.init(project='VanillaSGM',
                config={
                    'dataset': args.dataset,
                    'batch_size': args.batch_size,
                    'n_epochs': args.n_epochs,
                    'lr': args.lr,
                    'sigma': args.sigma,
                    'model_channels': args.model_channels,
                    'embed_dim': args.embed_dim
                },
                name = 'VanillaSGM_{}'.format(args.dataset))
    dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize)
    model = VanillaSGM(device, args.sigma, args.n_epochs, args.lr, args.model_channels, args.embed_dim, channels, input_size, args.dataset, args.sample_and_save_freq, args.sample_timesteps, args.snr, args.sampler_type, args.atol, args.rtol, args.eps)
    model.train_model(dataloader)
    wandb.finish()

elif args.sample:
    _, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize)
    model = VanillaSGM(device, args.sigma, args.n_epochs, args.lr, args.model_channels, args.embed_dim, channels, input_size, args.dataset, args.sample_and_save_freq, args.sample_timesteps, args.snr, args.sampler_type, args.atol, args.rtol, args.eps)
    model.model.load_state_dict(torch.load(args.checkpoint))
    model.sample(args.num_samples)

elif args.outlier_detection:
    dataloader_a, input_size_a, channels_a = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize)
    dataloader_b, input_size_b, channels_b = pick_dataset(args.out_dataset, 'val', args.batch_size, normalize=normalize)
    model = VanillaSGM(device, args.sigma, args.n_epochs, args.lr, args.model_channels, args.embed_dim, channels_a, input_size_a, args.dataset, args.sample_and_save_freq, args.sample_timesteps, args.snr, args.sampler_type, args.atol, args.rtol, args.eps)
    model.model.load_state_dict(torch.load(args.checkpoint))
    model.outlier_detection(dataloader_a, dataloader_b)

else:
    raise ValueError('Please specify at least one of the following: train, sample, outlier_detection')
