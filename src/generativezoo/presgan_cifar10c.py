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

class CIFAR10C(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise', severity=5):
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.array = np.load(os.path.join(data_raw_dir, 'cifar10c', f'{corruption}.npy'))[10000*(severity-1):10000*(severity)]

    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, idx):
        img = self.array[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, 0
    
def create_corrupted_loader(corruption, severity, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10C(root=data_raw_dir, transform=transform, corruption=corruption, severity=severity)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def cifar_test_loader(batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    validation_data = datasets.CIFAR10(root=data_raw_dir, train=False, download=True, transform=transform)
    
    test_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    
    return test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corruption_types = os.listdir(os.path.join(data_raw_dir, 'cifar10c'))
# remove the one saying labels
corruption_types.remove('labels.npy')
#remove all the .npy
corruption_types = [i.replace('.npy', '') for i in corruption_types]
corruption_types.sort()

# create a dataframe to store the results with corruption_types*5 rows
results = pd.DataFrame(columns=['corruption_type', 'AUROC', 'FPR95', 'Mean Scores'], index=range(len(corruption_types)*5 + 1))

# pre-fill the dataframe with the corruption types and all other columns as 0.0 without using append
counter = 0
for i in corruption_types:
    for j in range(1, 6):
        results.loc[counter] = ({'corruption_type': f'{i}_{j}', 'AUROC': 0.0, 'FPR95': 0.0, 'Mean Scores': 0.0})
        counter += 1
results.loc[counter] = ({'corruption_type': 'mean', 'AUROC': 0.0, 'FPR95': 0.0, 'Mean Scores': 0.0})

args = parse_args_PresGAN()

args.discriminator_checkpoint = "./../../models/PrescribedGAN/PresDisc_cifar10_1024.pt"
args.sigma_checkpoint = "./../../models/PrescribedGAN/PresSigma_cifar10_1024.pt"
args.nz = 1024
input_size = 32
channels = 3
args.ngf = 64
args.ndf = 64

in_loader = cifar_test_loader(batch_size=args.batch_size)

model = PresGAN(imgSize=input_size, nz=args.nz, ngf = args.ngf, ndf = args.ndf, nc = channels, device = device, beta1 = args.beta1, lrD = args.lrD, lrG = args.lrG, n_epochs = args.n_epochs, sigma_lr=args.sigma_lr,
                    num_gen_images=args.num_gen_images, restrict_sigma=args.restrict_sigma, sigma_min=args.sigma_min, sigma_max=args.sigma_max, stepsize_num=args.stepsize_num, lambda_=args.lambda_,
                    burn_in=args.burn_in, num_samples_posterior=args.num_samples_posterior, leapfrog_steps=args.leapfrog_steps, hmc_learning_rate=args.hmc_learning_rate, hmc_opt_accept=args.hmc_opt_accept, flag_adapt=args.flag_adapt, 
                    sample_and_save_freq=args.sample_and_save_freq, dataset=args.dataset)
model.load_checkpoints(generator_checkpoint=args.checkpoint, discriminator_checkpoint=args.discriminator_checkpoint, sigma_checkpoint=args.sigma_checkpoint)

auroc_list = []
fpr95_list = []
in_array = None

for corruption in corruption_types:
    for i in range(1, 6):
        out_loader = create_corrupted_loader(corruption, i, args.batch_size)
        auroc, fpr95, in_array, scores = model.outlier_detection(in_loader, out_loader, in_array=in_array, display=False)
        auroc_list.append(auroc)
        fpr95_list.append(fpr95)
        results.loc[results['corruption_type'] == f'{corruption}_{i}', 'AUROC'] = auroc
        results.loc[results['corruption_type'] == f'{corruption}_{i}', 'FPR95'] = fpr95
        results.loc[results['corruption_type'] == f'{corruption}_{i}', 'Mean Scores'] = scores
        print(f"Corruption: {corruption}, Severity: {i}, AUROC: {auroc:.4f}")

results.loc[results['corruption_type'] == 'mean', 'AUROC'] = np.mean(auroc_list)
results.loc[results['corruption_type'] == 'mean', 'FPR95'] = np.mean(fpr95_list)
results.loc[results['corruption_type'] == 'mean', 'Mean Scores'] = np.mean(in_array)
# print mean
auroc_list = np.array(auroc_list)
print(np.mean(auroc_list))

results.to_csv('cifar10c_results_presgan.csv', index=False)