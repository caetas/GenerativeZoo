import torch
from models.GAN.PrescribedGAN import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import os
from PIL import Image
from config import data_raw_dir
from tqdm import tqdm
import pandas as pd
from utils.util import parse_args_PresGAN
from data.Dataloaders import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modes = [i for i in range(0, 17)]

# create a dataframe to store the results with corruption_types*5 rows
results = pd.DataFrame(columns=['mode', 'AUROC', 'FPR95', 'Mean Scores'], index=range(len(modes) + 1))

# pre-fill the dataframe with the corruption types and all other columns as 0.0 without using append
counter = 0
for i in modes:
    results.loc[counter] = ({'mode': f'{i}', 'AUROC': 0.0, 'FPR95': 0.0, 'Mean Scores': 0.0})
    counter += 1
results.loc[counter] = ({'mode': 'mean', 'AUROC': 0.0, 'FPR95': 0.0, 'Mean Scores': 0.0})

args = parse_args_PresGAN()

args.discriminator_checkpoint = "./../../models/PrescribedGAN/PresDisc_xray_1024.pt"
args.sigma_checkpoint = "./../../models/PrescribedGAN/PresSigma_xray_1024.pt"
args.nz = 1024
input_size = 128
channels = 3

in_loader, input_size, channels = xrays_test_loader(batch_size=args.batch_size, input_shape=128, flavour=17, normalize=True)

model = PresGAN(imgSize=input_size, nz=args.nz, ngf = args.ngf, ndf = args.ndf, nc = channels, device = device, beta1 = args.beta1, lrD = args.lrD, lrG = args.lrG, n_epochs = args.n_epochs, sigma_lr=args.sigma_lr,
                    num_gen_images=args.num_gen_images, restrict_sigma=args.restrict_sigma, sigma_min=args.sigma_min, sigma_max=args.sigma_max, stepsize_num=args.stepsize_num, lambda_=args.lambda_,
                    burn_in=args.burn_in, num_samples_posterior=args.num_samples_posterior, leapfrog_steps=args.leapfrog_steps, hmc_learning_rate=args.hmc_learning_rate, hmc_opt_accept=args.hmc_opt_accept, flag_adapt=args.flag_adapt, 
                    sample_and_save_freq=args.sample_and_save_freq, dataset=args.dataset)
model.load_checkpoints(generator_checkpoint=args.checkpoint, discriminator_checkpoint=args.discriminator_checkpoint, sigma_checkpoint=args.sigma_checkpoint)

in_array = None
auroc_list = []
fpr95_list = []

for i in modes:
    out_loader, _, _ = xrays_test_loader(batch_size=args.batch_size, input_shape=128, flavour=i, normalize=True)
    auroc, fpr95, in_array, scores = model.outlier_detection(in_loader, out_loader, in_array=in_array, display=False)
    auroc_list.append(auroc)
    fpr95_list.append(fpr95)
    results.loc[results['mode'] == f'{i}', 'AUROC'] = auroc
    results.loc[results['mode'] == f'{i}', 'FPR95'] = fpr95
    results.loc[results['mode'] == f'{i}', 'Mean Scores'] = scores
    print(f"Mode: {i}, AUROC: {auroc:.4f}")

results.loc[results['mode'] == 'mean', 'AUROC'] = np.mean(auroc_list)
results.loc[results['mode'] == 'mean', 'FPR95'] = np.mean(fpr95_list)
results.loc[results['mode'] == 'mean', 'Mean Scores'] = np.mean(in_array)

print(f'Mean AUROC: {np.mean(auroc_list):.4f}')

# Save the results
results.to_csv('results_xray_presgan.csv', index=False)