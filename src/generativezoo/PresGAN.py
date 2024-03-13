from models.GAN.PrescribedGAN import *
from data.Dataloaders import *
from utils.util import parse_args_PresGAN
import torch

args = parse_args_PresGAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.train:
    train_dataloader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size = 32)
    model = PresGAN(imgSize=input_size, nz=args.nz, ngf = args.ngf, ndf = args.ndf, nc = channels, device = device, beta1 = args.beta1, lrD = args.lrD, lrG = args.lrG, n_epochs = args.n_epochs, sigma_lr=args.sigma_lr,
                    num_gen_images=args.num_gen_images, restrict_sigma=args.restrict_sigma, sigma_min=args.sigma_min, sigma_max=args.sigma_max, stepsize_num=args.stepsize_num, lambda_=args.lambda_,
                    burn_in=args.burn_in, num_samples_posterior=args.num_samples_posterior, leapfrog_steps=args.leapfrog_steps, hmc_learning_rate=args.hmc_learning_rate, hmc_opt_accept=args.hmc_opt_accept, flag_adapt=args.flag_adapt)
    model.train_model(train_dataloader)