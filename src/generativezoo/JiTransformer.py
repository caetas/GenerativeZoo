from models.FM.JiT import Denoiser
from utils.util import parse_args_JiT
from data.Dataloaders import pick_dataset

if __name__ == "__main__":

    args = parse_args_JiT()

    if args.train:

        train_loader, in_shape, in_channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, size = args.img_size, num_workers=args.num_workers)
        model = Denoiser(args)
        model.train_model(train_loader)