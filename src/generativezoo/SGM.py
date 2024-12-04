from models.SM.SGM import *
from data.Dataloaders import *
from utils.util import parse_args_SGM
import torch
import wandb

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args_SGM()
    normalize = True


    if args.train:
        dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=args.size, num_workers=args.num_workers)
        model = SGM(args, channels, input_size)
        model.train_model(dataloader)

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize, size=args.size)
        model = SGM(args, channels, input_size)
        model.model.load_state_dict(torch.load(args.checkpoint))
        model.sample(args.num_samples)

    else:
        raise ValueError('Please specify at least one of the following: train, sample, outlier_detection')

