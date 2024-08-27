from models.GAN.PrescribedGAN import *
from data.Dataloaders import *
from utils.util import parse_args_PresGAN
import torch
import wandb

if __name__ == '__main__':
    args = parse_args_PresGAN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = None

    if args.train:
        train_dataloader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size=size, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project="PresGAN",
                    
                    config = {
                            "dataset": args.dataset,
                            "batch_size": args.batch_size,
                            "nz": args.nz,
                            "ngf": args.ngf,
                            "ndf": args.ndf,
                            "lrD": args.lrD,
                            "lrG": args.lrG,
                            "beta1": args.beta1,
                            "n_epochs": args.n_epochs,
                            "sigma_lr": args.sigma_lr,
                            "num_gen_images": args.num_gen_images,
                            "restrict_sigma": args.restrict_sigma,
                            "sigma_min": args.sigma_min,
                            "sigma_max": args.sigma_max,
                            "stepsize_num": args.stepsize_num,
                            "lambda_": args.lambda_,
                            "burn_in": args.burn_in,
                            "num_samples_posterior": args.num_samples_posterior,
                            "leapfrog_steps": args.leapfrog_steps,
                            "hmc_learning_rate": args.hmc_learning_rate,
                            "hmc_opt_accept": args.hmc_opt_accept,
                            "flag_adapt": args.flag_adapt
                    },

                    name=f"PresGAN_{args.dataset}"

                    )
        model = PresGAN(imgSize=input_size, channels=channels, args=args)
        model.train_model(train_dataloader)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size = size)
        model = PresGAN(imgSize=input_size, channels=channels, args=args)
        model.load_checkpoints(generator_checkpoint=args.checkpoint, discriminator_checkpoint=args.discriminator_checkpoint, sigma_checkpoint=args.sigma_checkpoint)
        model.sample(num_samples=args.num_gen_images)

    elif args.outlier_detection:
        in_loader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size = size, mode="val")
        out_loader, _, _ = pick_dataset(dataset_name = args.out_dataset, batch_size=args.batch_size, normalize = True, size = input_size, mode="val")
        model = PresGAN(imgSize=input_size, channels=channels, args=args)
        model.load_checkpoints(generator_checkpoint=args.checkpoint, discriminator_checkpoint=args.discriminator_checkpoint, sigma_checkpoint=args.sigma_checkpoint)
        model.outlier_detection(in_loader, out_loader)

    else:
        raise Exception("Invalid mode. Set the --train or --sample flag")