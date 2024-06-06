from models.SGM.NCSNPlusPlus import *
from utils.util import parse_args_NCSNPlusPlus
import torch
from data.Dataloaders import *
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parse_args_NCSNPlusPlus()

if args.dataset == "mnist" or args.dataset == "fashionmnist":
    size = 32
else:
    size = None

if args.train:
    train_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=False, size=size)
    wandb.init(project="NCSNPlusPlus",
               
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
                        "ch_mult": args.ch_mult,
                        "num_res_blocks": args.num_res_blocks,
                        "attn_resolutions": args.attn_resolutions,
                        "resamp_with_conv": args.resamp_with_conv,
                        "conditional": args.conditional,
                        "fir": args.fir,
                        "fir_kernel": args.fir_kernel,
                        "skip_rescale": args.skip_rescale,
                        "resblock_type": args.resblock_type,
                        "progressive": args.progressive,
                        "progressive_input": args.progressive_input,
                        "progressive_combine": args.progressive_combine,
                        "attn_type": args.attn_type,
                        "embedding_type": args.embedding_type,
                        "init_scale": args.init_scale,
                        "fourier_scale": args.fourier_scale,
                        "conv_size": args.conv_size
                    },

                name=f"NCSNPlusPlus_{args.dataset}"
                    
                    )

    model = NCSNPlusPlus(args,input_size, channels)
    print(model)
    model.train_model(train_loader, args)
    wandb.finish()

elif args.sample:
    _, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=False, size=size)
    model = NCSNPlusPlus(args,input_size, channels)
    if args.checkpoint is not None:
        model.load_checkpoints(args.checkpoint)
    model.sample(args, False)

else:
    raise ValueError("Invalid mode, choose either train, sample or outlier_detection.")