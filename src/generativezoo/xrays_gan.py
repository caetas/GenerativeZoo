import torch
from models.GAN.VanillaGAN import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import os
from PIL import Image
from config import data_raw_dir
from tqdm import tqdm
import pandas as pd
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

latent_dim = 1024
d = 64
#hidden_dims_list = [[64, 128, 256]]
batch_size = 256
lr = 0.0002
checkpoints_base = './../../models/VanillaGAN'
discriminator_list = ['VanDisc_xray.pt']

in_loader, input_size, channels = xrays_test_loader(batch_size=batch_size, input_shape=128, flavour=17, normalize=True)

for discriminator_checkpoint in discriminator_list:
    model = Discriminator(channels=1, d=d, imgSize=input_size).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoints_base,discriminator_checkpoint)))
    auroc_list = []
    fpr95_list = []
    pbar = tqdm(range(17))
    in_array = None
    for c in pbar:
        # add postfix to corruption type
        corrupted_loader, _, _ = xrays_test_loader(batch_size=batch_size, input_shape=128, flavour=c, normalize=True)
        auroc, fpr95, in_array, scores = model.outlier_detection(in_loader, corrupted_loader, in_array=in_array, display=False, device=device)
        # add to dataframe

        auroc_list.append(auroc)
        fpr95_list.append(fpr95)
        results.loc[results['mode'] == f'{c}', 'AUROC'] = auroc
        results.loc[results['mode'] == f'{c}', 'FPR95'] = fpr95
        results.loc[results['mode'] == f'{c}', 'Mean Scores'] = scores

        pbar.set_description(f"Mode: {c}, AUROC: {auroc:.4f}")
    # add mean to dataframe

    results.loc[results['mode'] == 'mean', 'AUROC'] = np.mean(auroc_list)
    results.loc[results['mode'] == 'mean', 'FPR95'] = np.mean(fpr95_list)
    results.loc[results['mode'] == 'mean', 'Mean Scores'] = np.mean(in_array)
        
# save results as csv
results.to_csv('xrays_results_gan.csv', index=False)