import torch
from data.Dataloaders import *
from models.DDPM.VanillaDDPM import *
from utils.util import parse_args_DDPM
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args_DDPM()
normalize = True

if args.dataset == 'mnist':
     size = 32
else:
     size = None

if args.train:
     dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=size)
     model = VanillaDDPM(device, args, n_features=args.n_features, init_channels=args.init_channels, channel_scale_factors=args.channel_scale_factors, in_channels=channels, image_size=input_size)
     
     wandb.init(project='DDPM',
                
                config={
                    'dataset': args.dataset,
                    'batch_size': args.batch_size,
                    'n_epochs': args.n_epochs,
                    'lr': args.lr,
                    'timesteps': args.timesteps,
                    'beta_start': args.beta_start,
                    'beta_end': args.beta_end,
                    'ddpm': args.ddpm,
                    'input_size': input_size,
                    'channels': channels,
                    'loss_type': args.loss_type,
                    'channel_scale_factors': args.channel_scale_factors,
                    'init_channels': args.init_channels,
                    'n_features': args.n_features
                },

                name = 'DDPM_{}'.format(args.dataset))
     
     model.train_model(dataloader)
     wandb.finish()

elif args.sample:
     _, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize, size=size)
     model = VanillaDDPM(device, args, n_features=args.n_features, init_channels=args.init_channels, channel_scale_factors=args.channel_scale_factors, in_channels=channels, image_size=input_size)
     model.denoising_model.load_state_dict(torch.load(args.checkpoint))
     model.sample(args.n_samples)

elif args.outlier_detection:
     dataloader_a, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize, size=size)
     model = VanillaDDPM(device, args, n_features=args.n_features, init_channels=args.init_channels, channel_scale_factors=args.channel_scale_factors, in_channels=channels, image_size=input_size)
     model.denoising_model.load_state_dict(torch.load(args.checkpoint))
     dataloader_b, input_size_b, channels_b = pick_dataset(args.out_dataset, 'val', args.batch_size, normalize=normalize, good = False, size=input_size)
     model.outlier_detection(dataloader_a,dataloader_b, args.dataset, args.out_dataset)

else:
     raise ValueError('Please specify at least one of the following: train, sample, outlier_detection')