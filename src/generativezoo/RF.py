from data.Dataloaders import pick_dataset
from models.Flow.RectifiedFlows import RF
from utils.util import parse_args_RectifiedFlows
import wandb

if __name__ == '__main__':

    args = parse_args_RectifiedFlows()


    if args.train:
        if not args.no_wandb:
            wandb.init(project='RectifiedFlows',
                        config={
                            "dataset": args.dataset,
                            "batch_size": args.batch_size,
                            "n_epochs": args.n_epochs,
                            "lr": args.lr,
                            "patch_size": args.patch_size,
                            "dim": args.dim,
                            "n_layers": args.n_layers,
                            "n_heads": args.n_heads,
                            "multiple_of": args.multiple_of,
                            "ffn_dim_multiplier": args.ffn_dim_multiplier,
                            "norm_eps": args.norm_eps,
                            "class_dropout_prob": args.class_dropout_prob,
                            "num_classes": args.num_classes,
                        },

                        name=f"RectifiedFlows_{args.dataset}")

        train_loader, input_size, channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, num_workers=args.num_workers)
        model = RF(args, input_size, channels)
        model.train_model(train_loader)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(args.dataset, batch_size = 1, normalize=True)
        model = RF(args, input_size, channels)
        model.load_checkpoint(args.checkpoint)
        model.sample(16)