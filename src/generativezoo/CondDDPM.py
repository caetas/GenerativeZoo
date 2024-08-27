import torch
from data.Dataloaders import *
from models.DDPM.ConditionalDDPM import *
from utils.util import parse_args_CDDPM
import wandb

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args_CDDPM()
    normalize = True

    size = None

    if args.train:
        train_dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=size, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project='CDDPM',
                        config={
                            'dataset': args.dataset,
                            'batch_size': args.batch_size,
                            'n_epochs': args.n_epochs,
                            'lr': args.lr,
                            'n_features': args.n_features,
                            'n_classes': args.n_classes,
                            'drop_prob': args.drop_prob,
                            'timesteps': args.timesteps,
                            'beta_start': args.beta_start,
                            'beta_end': args.beta_end,
                            'ddpm': args.ddpm,
                            'input_size': input_size,
                            'channels': channels,
                        },

                        name = 'CDDPM_{}'.format(args.dataset))
        model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=args)
        model.train_model(train_dataloader)

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=size)
        model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=args)
        model.denoising_model.load_state_dict(torch.load(args.checkpoint))
        model.sample(guide_w=args.guide_w)

    elif args.outlier_detection:
        pass

    else:
        raise ValueError('Please specify at least one of the following: train, sample, outlier_detection')