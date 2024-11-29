import torch
from data.Dataloaders import *
from models.DDPM.ConditionalDDPM import *
from utils.util import parse_args_CDDPM
import wandb

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args_CDDPM()
    normalize = True

    if args.train:
        train_dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=args.size, num_workers=args.num_workers)
        model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=args)
        model.train_model(train_dataloader)

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=args.size)
        model = ConditionalDDPM(in_channels=channels, input_size=input_size, args=args)
        model.denoising_model.load_state_dict(torch.load(args.checkpoint))
        model.sample()

    else:
        raise ValueError('Please specify at least one of the following: train, sample')