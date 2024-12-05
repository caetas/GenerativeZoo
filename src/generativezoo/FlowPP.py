from models.NF.FlowPlusPlus import *
from data.Dataloaders import *
from utils.util import parse_args_FlowPP
import wandb

if __name__ == '__main__':

    args = parse_args_FlowPP()

    size = None

    if args.train:
        train_loader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=False, size=size, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project='FlowPlusPlus',
                    config={
                        'dataset': args.dataset,
                        'batch_size': args.batch_size,
                        'n_epochs': args.n_epochs,
                        'warm_up': args.warm_up,
                        'lr': args.lr,
                        'grad_clip': args.grad_clip,
                        'num_blocks': args.num_blocks,
                        'num_components': args.num_components,
                        'num_channels': args.num_channels,
                        'use_attn': args.use_attn,
                        'num_dequant_blocks': args.num_dequant_blocks,
                        'drop_prob': args.drop_prob,
                        },
                        name = 'FlowPlusPlus_{}'.format(args.dataset))
        model = FlowPlusPlus(args, channels=channels, img_size=input_size)
        model.train_model(args, train_loader)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=False, size=size)
        model = FlowPlusPlus(args, channels=channels, img_size=input_size)
        model.load_checkpoints(args)
        model.sample(16, False)

    elif args.outlier_detection:
        in_loader, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=False, size=size)
        out_loader, _, _ = pick_dataset(args.dataset, 'val', args.batch_size, normalize=False, size=input_size)
        model = FlowPlusPlus(args, channels=channels, img_size=input_size)
        model.load_checkpoints(args)
        model.outlier_detection(in_loader, out_loader)