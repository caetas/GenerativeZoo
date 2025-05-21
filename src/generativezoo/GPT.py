from models.AR.GPT import VQGAN_GPT
from utils.util import parse_args_GPT
from data.Dataloaders import *

if __name__ == "__main__":
    args = parse_args_GPT()

    if args.train:
        train_loader, input_size, channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        val_loader, _, _ = pick_dataset(args.dataset, mode='val', batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        model = VQGAN_GPT(args, channels, input_size)
        model.train_model(train_loader, val_loader)

    if args.sample:
        _, input_size, channels = pick_dataset(args.dataset, mode='val', batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        model = VQGAN_GPT(args, channels, input_size)
        model.load_checkpoint(args.checkpoint_gpt)
        model.sample()