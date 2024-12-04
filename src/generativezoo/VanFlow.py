from models.NF.VanillaFlow import VanillaFlow
from utils.util import parse_args_VanillaFlow
from data.Dataloaders import *
import wandb

if __name__ == '__main__':

    args = parse_args_VanillaFlow()

    if args.train:
        in_loader, img_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project = "VanillaFlow",
                    config = {
                            "dataset": args.dataset,
                            "batch_size": args.batch_size,
                            "epochs": args.n_epochs,
                            "lr": args.lr,
                            "img_size": img_size,
                            "c_hidden": args.c_hidden,
                            "n_layers": args.n_layers,
                            "multi_scale": args.multi_scale,
                            "vardeq": args.vardeq,
                    },
                    name = f"VanillaFlow_{args.dataset}")
        
        model = VanillaFlow(img_size, channels, args)
        model.train_model(in_loader, args)
        wandb.finish()

    elif args.sample:
        _, img_size, channels = pick_dataset(args.dataset, 'val', args.batch_size)
        model = VanillaFlow(img_size, channels, args)
        if args.checkpoint is not None:
            model.flows.load_state_dict(torch.load(args.checkpoint))
        model.sample(train=False)

    elif args.outlier_detection:
        in_loader, img_size, channels = pick_dataset(args.dataset, 'val', args.batch_size)
        out_loader, _, _ = pick_dataset(args.dataset, 'val', args.batch_size, size=img_size)
        model = VanillaFlow(img_size, channels, args)
        if args.checkpoint is not None:
            model.flows.load_state_dict(torch.load(args.checkpoint))
        model.outlier_detection(in_loader, out_loader)