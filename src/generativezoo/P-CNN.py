from models.AR.PixelCNN import *
from data.Dataloaders import *
from utils.util import parse_args_PixelCNN
import wandb

args = parse_args_PixelCNN()

if args.dataset == "mnist" or args.dataset == "fashionmnist":
    size = 32
else:
    size = None

if args.train:
    dataloader, img_size, channels = pick_dataset(args.dataset, normalize=False, batch_size=args.batch_size, size=size)

    wandb.init(project="PixelCNN",
               config = {
                     "batch_size": args.batch_size,
                     "hidden_channels": args.hidden_channels,
                     "n_epochs": args.n_epochs,
                     "lr": args.lr,
                     "gamma": args.gamma,
                     "image_size": img_size,
                     "dataset": args.dataset,
                     "channels": channels
               },
               name=f"PixelCNN_{args.dataset}"
               )

    model = PixelCNN(channels, args.hidden_channels)
    model.train_model(dataloader, args, img_size)
    wandb.finish()

elif args.sample:
    _, img_size, channels = pick_dataset(args.dataset, normalize=False, batch_size=args.batch_size, size=size)
    model = PixelCNN(channels, args.hidden_channels)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    model.sample((16,channels,img_size,img_size), train=False)