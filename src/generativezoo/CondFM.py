from models.Flow.CondFlowMatching import CondFlowMatching
from data.Dataloaders import *
from utils.util import parse_args_CondFlowMatching
import wandb

if __name__ == '__main__':

    args = parse_args_CondFlowMatching()

    if args.train:
        train_loader, input_size, channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project='CondFlowMatching',
                        config={
                            "dataset": args.dataset,
                            "batch_size": args.batch_size,
                            "n_epochs": args.n_epochs,
                            "lr": args.lr,
                            "channels": channels,
                            "input_size": input_size,
                            'model_channels': args.model_channels,
                            'num_res_blocks': args.num_res_blocks,
                            'attention_resolutions': args.attention_resolutions,
                            'dropout': args.dropout,
                            'channel_mult': args.channel_mult,
                            'conv_resample': args.conv_resample,
                            'dims': args.dims,
                            'num_heads': args.num_heads,
                            'num_head_channels': args.num_head_channels,
                            'use_scale_shift_norm': args.use_scale_shift_norm,
                            'resblock_updown': args.resblock_updown,
                            'use_new_attention_order': args.use_new_attention_order,
                            "dropout_prob": args.dropout_prob,
                            "cfg": args.cfg,
                        },

                        name=f"CondFlowMatching_{args.dataset}")    
        model = CondFlowMatching(args, input_size, channels)
        model.train_model(train_loader)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, batch_size = 1, normalize=True)
        model = CondFlowMatching(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.sample(args.num_samples, train=False)
    else:
        raise ValueError("Invalid mode, please specify train or sample mode.")
