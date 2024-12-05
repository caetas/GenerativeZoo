from models.NF.Glow import *
from data.Dataloaders import *
from utils.util import parse_args_Glow
import wandb


if __name__ == '__main__':

    args = parse_args_Glow()
    normalize = False

    size = None

    if args.train:
        if not args.no_wandb:
            wandb.init(project='GLOW',
                    config={
                            "batch_size": args.batch_size,
                            "lr": args.lr,
                            "n_epochs": args.n_epochs,
                            "dataset": args.dataset,
                            "hidden_channels": args.hidden_channels,
                            "K": args.K,
                            "L": args.L,
                            "actnorm_scale": args.actnorm_scale,
                            "flow_permutation": args.flow_permutation,
                            "flow_coupling": args.flow_coupling,
                            "LU_decomposed": args.LU_decomposed,
                            "learn_top": args.learn_top,
                            "y_condition": args.y_condition,
                            "num_classes": args.num_classes,
                            "n_bits": args.n_bits,  
                    },

                        name = 'GLOW_{}'.format(args.dataset))
        
        train_loader, input_shape, channels = pick_dataset(args.dataset, batch_size=args.batch_size, normalize=normalize, size=size, num_workers=args.num_workers)
        model = Glow(image_shape        =   (input_shape,input_shape,channels), hidden_channels    =   args.hidden_channels, args=args)
        model.train_model(train_loader, args)

    elif args.sample:
        _, input_shape, channels = pick_dataset(args.dataset, batch_size=args.batch_size, normalize=normalize, size=size, num_workers=0)
        model = Glow(image_shape        =   (input_shape,input_shape,channels), hidden_channels    =   args.hidden_channels, args=args)
        model.load_checkpoint(args)
        model.sample(train=False)

    elif args.outlier_detection:
        in_loader, input_shape, channels = pick_dataset(args.dataset, batch_size=args.batch_size, normalize=normalize, size=size, num_workers=0, mode='val')
        out_loader, _, _ = pick_dataset(args.out_dataset, batch_size=args.batch_size, normalize=normalize, size=input_shape, num_workers=0, mode='val')
        model = Glow(image_shape        =   (input_shape,input_shape,channels), hidden_channels    =   args.hidden_channels, args=args)
        model.load_checkpoint(args)
        model.outlier_detection(in_loader, out_loader)

    else:
        raise ValueError("Invalid mode. Please specify train or sample")
