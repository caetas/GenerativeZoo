from models.VAE.HierarchicalVAE import *
from data.Dataloaders import *
from utils.util import parse_args_HierarchicalVAE
import torch
import wandb

args = parse_args_HierarchicalVAE()

size = None

if args.train:
    dataloader, img_size, channels = pick_dataset(args.dataset, size=size, batch_size=args.batch_size)
    if not args.no_wandb:
        wandb.init(project='HierarchicalVAE',
                config={
                        'latent_dim': args.latent_dim,
                        'img_size': img_size,
                        'channels': channels,
                        'batch_size': args.batch_size,
                        'epochs': args.n_epochs,
                        'dataset': args.dataset
                    },
                    name=f'HierarchicalVAE_{args.dataset}')

    model = HierarchicalVAE(args.latent_dim, (img_size, img_size), channels, args.no_wandb)
    model.train_model(dataloader, args)
    wandb.finish()

if args.sample:
    model = HierarchicalVAE(args.latent_dim, (size, size), channels)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
    model.sample(args)