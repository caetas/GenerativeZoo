from models.FM.CondFlowMatching import CondFlowMatching
from data.Dataloaders import *
from utils.util import parse_args_CondFlowMatching
import wandb

if __name__ == '__main__':

    args = parse_args_CondFlowMatching()

    if args.train:
        train_loader, input_size, channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size) 
        model = CondFlowMatching(args, input_size, channels)
        model.train_model(train_loader)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, batch_size = 1, normalize=True, size=args.size)
        model = CondFlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.sample(args.num_samples, train=False)
    elif args.fid:
        _, input_size, channels = pick_dataset(args.dataset, batch_size = 1, normalize=True, size=args.size)
        model = CondFlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.fid_sample()

    elif args.translation:
        val_loader, input_size, channels = pick_dataset(args.dataset, batch_size = 16, normalize=True, size=args.size, mode='val')
        model = CondFlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.image_translation(val_loader)
    else:
        raise ValueError("Invalid mode, please specify train or sample mode.")
