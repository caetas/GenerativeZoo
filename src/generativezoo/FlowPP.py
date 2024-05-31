from models.Flow.FlowPlusPlus import *
from data.Dataloaders import *
from utils.util import parse_args_FlowPP
import wandb

args = parse_args_FlowPP()

if args.dataset == 'mnist':
    size = 32
else:
    size = None

if args.train:
    train_loader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=False, size=size)
    wandb.init(project='FlowPlusPlus',
               config={
                   'dataset': args.dataset,
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'warm_up': args.warm_up,
                   'lr': args.lr,
                   'sample_and_save_frequency': args.sample_and_save_frequency,
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
    model = FlowPlusPlus(args)
    model.sample(16)