from models.GAN.WGAN import *
import torch
from data.Dataloaders import *
import wandb
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils.util import parse_args_WassersteinGAN

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args_WassersteinGAN()

    size = None

    if args.train:
        if not args.no_wandb:
            wandb.init(project="WGAN",
                    config={
                        "dataset": args.dataset,
                        "batch_size": args.batch_size,
                        "n_epochs": args.n_epochs,
                        "latent_dim": args.latent_dim,
                        "d": args.d,
                        "lrg": args.lrg,
                        "lrd": args.lrd,
                        "beta1": args.beta1,
                        "beta2": args.beta2,
                        "n_critic": args.n_critic,
                        "gp_weight": args.gp_weight
                    },
                    name=f"WGAN_{args.dataset}")
        
        train_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=False, size=size, num_workers=args.num_workers)
        model = WGAN(args=args, imgSize=input_size, channels=channels)
        model.train_model(train_loader)
        wandb.finish()

    elif args.sample:
        _, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=1, normalize=False, size=size)
        model = Generator(latent_dim=args.latent_dim, channels=channels, d=args.d, imgSize=input_size).to(device)
        model.load_state_dict(torch.load(args.checkpoint))
        model.sample(n_samples=args.n_samples, device=device)

    elif args.outlier_detection:

        in_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=False, size=size, mode='val')
        out_loader, _, _ = pick_dataset(dataset_name=args.out_dataset, batch_size=args.batch_size, normalize=False, size=input_size, mode='val')

        model = WGAN(batch_size = args.batch_size, latent_dim=args.latent_dim, d=args.d, lrg=args.lrg, lrd=args.lrd, beta1=args.beta1, beta2=args.beta2, gp_weight=args.gp_weight, dataset=args.dataset, n_epochs=args.n_epochs, n_critic=args.n_critic, sample_and_save_freq=args.sample_and_save_freq, imgSize=input_size, channels=channels)
        model.D.load_state_dict(torch.load(args.discriminator_checkpoint))
        model.outlier_detection(in_loader, out_loader, display=True)