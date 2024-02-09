from models.GANs.VanillaGAN import *
from data.Dataloaders import *
from utils.util import parse_args_VanillaGAN
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import subprocess
import wandb

args = parse_args_VanillaGAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.train:
    subprocess.run(['wandb', 'login'])
    wandb.init(project='VanillaGAN',
               config={
                   'dataset': args.dataset,
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'latent_dim': args.latent_dim,
                   'd': args.d,
                   'lr': args.lr,
                   'beta1': args.beta1,
                   'beta2': args.beta2,
                   'sample_interval': args.sample_interval
               },
               name = 'VanillaGAN_{}'.format(args.dataset))
    train_dataloader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True)
    model = VanillaGAN(n_epochs = args.n_epochs, device = device, latent_dim = args.latent_dim, d = args.d, channels = channels, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2, img_size = input_size, sample_interval = args.sample_interval)
    model.train_model(train_dataloader)

elif args.sample:
    _, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=1, normalize = True)
    model = Generator(latent_dim = args.latent_dim, channels = channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    model.sample(n_samples = args.n_samples, device = device)
else:
    raise Exception('Please specify either --train or --sample')