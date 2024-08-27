from models.DDPM.MONAI_DiffAE import DiffAE
import torch
from data.Dataloaders import *
from utils.util import parse_args_DiffAE
import wandb


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args_DiffAE()

    size = None

    if args.train:
        train_dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=True, size=size, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project='DiffAE',
                        config={
                            'dataset': args.dataset,
                            'batch_size': args.batch_size,
                            'n_epochs': args.n_epochs,
                            'lr': args.lr,
                            'embedding_dim': args.embedding_dim,
                            'timesteps': args.timesteps,
                            'sample_timesteps': args.sample_timesteps,
                            'model_channels': args.model_channels,
                            'attention_levels': args.attention_levels,
                            'num_res_blocks': args.num_res_blocks,
                            'input_size': input_size,
                            'channels': channels,
                        },
                        name = 'DiffAE_{}'.format(args.dataset))
        model = DiffAE(args, channels)
        model.train_model(train_dataloader, train_dataloader)
        wandb.finish()

    elif args.manipulate:
        train_dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=True, size = size)
        val_dataloader, _, _ = pick_dataset(args.dataset, 'val', args.batch_size, normalize=True, size=size)
        model = DiffAE(args, channels)
        model.unet.load_state_dict(torch.load(args.checkpoint))
        model.linear_regression(train_dataloader, val_dataloader)
        model.manipulate_latent(val_dataloader)
