from models.FM.FlowMatching import FlowMatching
from data.Dataloaders import *
from utils.util import parse_args_FlowMatching
import wandb

if __name__ == '__main__':

    args = parse_args_FlowMatching()

    if args.train:
        train_loader, input_size, channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        model = FlowMatching(args, input_size, channels)
        model.train_model(train_loader)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, batch_size = 1, normalize=True, size=args.size)
        model = FlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.sample(args.num_samples, train=False)

    elif args.outlier_detection:
        in_loader, input_size, channels = pick_dataset(args.dataset, mode='val', batch_size = args.batch_size, normalize=True, size=args.size)
        out_loader, _, _ = pick_dataset(args.out_dataset, mode='val', batch_size = args.batch_size, normalize=True, size=input_size)
        model = FlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.outlier_detection(in_loader, out_loader)

    elif args.interpolation:
        in_loader, input_size, channels = pick_dataset(args.dataset, mode='val', batch_size = args.batch_size, normalize=True, size=args.size)
        model = FlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.interpolate(in_loader)
    elif args.fid:
        _, input_size, channels = pick_dataset(args.dataset, mode='val', batch_size = args.batch_size, normalize=True, size=args.size)
        model = FlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.fid_sample(args.batch_size)
    else:
        raise ValueError("Invalid mode, please specify train or sample mode.")
