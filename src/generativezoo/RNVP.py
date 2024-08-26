from models.Flow.RealNVP import RealNVP
from data.Dataloaders import *
from utils.util import parse_args_RealNVP
import wandb

if __name__ == '__main__':

    args = parse_args_RealNVP()

    size = None

    if args.train:
        dataloader, img_size, channels = pick_dataset(args.dataset, batch_size=args.batch_size, normalize = False, size=size, num_workers=args.num_workers)
        model = RealNVP(img_size=img_size, in_channels=channels, args=args)
        
        if not args.no_wandb:
            wandb.init(project='RealNVP',
                    
                    config = {"dataset": args.dataset,
                                "num_scales": args.num_scales,
                                "mid_channels": args.mid_channels,
                                "num_blocks": args.num_blocks,
                                "batch_size": args.batch_size,
                                "lr": args.lr,
                                "n_epochs": args.n_epochs,
                                "img_size": img_size,
                                "channels": channels},
                                
                                name=f"RealNVP_{args.dataset}")
        
        model.train_model(dataloader, args)
        wandb.finish()

    elif args.sample:
        _, img_size, channels = pick_dataset(args.dataset, batch_size=1, normalize = False, size=size)
        model = RealNVP(img_size=img_size, in_channels=channels, args=args)

        if args.checkpoint is not None:
            model.load_state_dict(torch.load(args.checkpoint))

        model.sample(16, train=False)

    elif args.outlier_detection:
        in_loader, img_size, channels = pick_dataset(args.dataset, mode = 'val', batch_size=args.batch_size, normalize = False, size=size)
        out_loader, _, _ = pick_dataset(args.out_dataset, mode = 'val', batch_size=args.batch_size, normalize = False, size=img_size)
        model = RealNVP(img_size=img_size, in_channels=channels, args=args)

        if args.checkpoint is not None:
            model.load_state_dict(torch.load(args.checkpoint))
        
        model.outlier_detection(in_loader, out_loader)

