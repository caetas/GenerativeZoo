from models.SM.NCSNv2 import *
from utils.util import parse_args_NCSNv2
import torch
from data.Dataloaders import *
import wandb


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args_NCSNv2()

    size = None

    if args.train:
        train_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=False, size=size, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project="NCSNv2",
                    
                        config = {
                                "dataset": args.dataset,
                                "batch_size": args.batch_size,
                                "n_steps": args.n_steps,
                                "lr": args.lr,
                                "n_epochs": args.n_epochs,
                                "beta1": args.beta1,
                                "beta2": args.beta2,
                                "weight_decay": args.weight_decay,
                                "nf": args.nf,
                                "snr": args.snr,
                                "probability_flow": args.probability_flow,
                                "predictor": args.predictor,
                                "corrector": args.corrector,
                                "noise_removal": args.noise_removal,
                                "sigma_max": args.sigma_max,
                                "sigma_min": args.sigma_min,
                                "num_scales": args.num_scales,
                                "normalization": args.normalization,
                                "continuous": args.continuous,
                                "reduce_mean": args.reduce_mean,
                                "likelihood_weighting": args.likelihood_weighting,
                                "act": args.act,
                            },

                        name=f"NCSNv2_{args.dataset}"
                            
                            )

        model = NCSNv2(input_size, channels, args)
        model.train_model(train_loader, args)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=False, size=size)
        model = NCSNv2(input_size, channels, args)
        model.load_checkpoints(args.checkpoint)
        model.sample(args, False)

    else:
        raise ValueError("Invalid mode, choose either train, sample or outlier_detection.")