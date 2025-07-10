from data.Dataloaders import pick_dataset
from models.FM.RectifiedFlows import RF
from utils.util import parse_args_RectifiedFlows

if __name__ == '__main__':

    args = parse_args_RectifiedFlows()


    if args.train:
        train_loader, input_size, channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, num_workers=args.num_workers, size=args.size)
        model = RF(args, input_size, channels)
        model.train_model(train_loader)

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, batch_size = 1, normalize=True, size=args.size)
        model = RF(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.sample(16)

    elif args.fid:
        _, input_size, channels = pick_dataset(args.dataset, batch_size = 1, normalize=True, size=args.size)
        model = RF(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.fid_sample()