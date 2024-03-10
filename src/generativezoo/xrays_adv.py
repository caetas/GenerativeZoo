from models.GAN.AdversarialVAE import *
from data.Dataloaders import *
import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 1024
hidden_dims_list = [[64, 128, 256, 512]]
batch_size = 256
lr = 0.0002
gen_weight = 1e-3
recon_weight = 1e-3
checkpoints_base = './../../models/AdversarialVAE'
checkpoints_list = ['AdvVAE_xray.pt']
discriminator_list = ['Discriminator_xray.pt']
loss_type = ['mse']
modes = list(range(17))

# create a dataframe to store the results with modess*5 rows
results = pd.DataFrame(columns=['modes', 'AUROC VAE', 'FPR95 VAE', 'Mean Score VAE', 'AUROC Discriminator', 'FPR95 Discriminator', 'Mean Score Discriminator'], index=range(len(modes) + 1))

# pre-fill the dataframe with the corruption types and all other columns as 0.0 without using append
counter = 0
for i in modes:
    results.loc[counter] = ({'modes': str(i), 'AUROC VAE': 0.0, 'FPR95 VAE': 0.0, 'Mean Score VAE':0.0 ,'AUROC Discriminator': 0.0, 'FPR95 Discriminator': 0.0, 'Mean Score Discriminator': 0.0})
    counter += 1
results.loc[counter] = ({'modes': 'mean', 'AUROC VAE': 0.0, 'FPR95 VAE': 0.0, 'Mean Score VAE':0.0 ,'AUROC Discriminator': 0.0, 'FPR95 Discriminator': 0.0, 'Mean Score Discriminator': 0.0})

for hidden_dims, checkpoint, discriminator_checkpoint in zip(hidden_dims_list, checkpoints_list, discriminator_list):
    for loss in loss_type:
        print(f"hidden_dims: {hidden_dims}, checkpoint: {checkpoint}, discriminator: {discriminator_checkpoint}, loss: {loss}")
        model = AdversarialVAE(input_shape = 128, device = device, input_channels = 1, latent_dim = latent_dim, n_epochs = 200, hidden_dims = hidden_dims.copy(), lr = lr, batch_size = batch_size, gen_weight = gen_weight, recon_weight=recon_weight, sample_and_save_frequency = 5)
        model.vae.load_state_dict(torch.load(os.path.join(checkpoints_base, checkpoint)))
        model.discriminator.load_state_dict(torch.load(os.path.join(checkpoints_base, discriminator_checkpoint)))
        in_loader, input_size, channels = xrays_test_loader(32, normalize=True, input_shape=128, flavour=17)
        aurocs = []
        aurocs_discriminator = []
        fpr95 = []
        fpr95_discriminator = []
        mean_scores = []
        mean_scores_discriminator = []
        pbar = tqdm(modes)
        in_array = None
        in_array_discriminator = None
        for i in pbar:
            # add postfix to corruption type
            out_loader, _, _ = xrays_test_loader(32, normalize=True, input_shape=128, flavour=i)
            in_array, in_array_discriminator, metrics = model.outlier_detection(in_loader, out_loader, display=False, in_array=in_array, in_array_discriminator=in_array_discriminator)
            aurocs.append(metrics['rocauc'])
            aurocs_discriminator.append(metrics['rocauc_d'])
            fpr95.append(metrics['fpr'])
            fpr95_discriminator.append(metrics['fpr_d'])
            mean_scores.append(metrics['mean'])
            mean_scores_discriminator.append(metrics['mean_d'])
            # add to dataframe

            results.loc[results['modes'] == str(i), 'AUROC VAE'] = metrics['rocauc']
            results.loc[results['modes'] == str(i), 'FPR95 VAE'] = metrics['fpr']
            results.loc[results['modes'] == str(i), 'Mean Score VAE'] = metrics['mean']
            results.loc[results['modes'] == str(i), 'AUROC Discriminator'] = metrics['rocauc_d']
            results.loc[results['modes'] == str(i), 'FPR95 Discriminator'] = metrics['fpr_d']
            results.loc[results['modes'] == str(i), 'Mean Score Discriminator'] = metrics['mean_d']
            pbar.set_description(f"mode: {i}, auroc: {metrics['rocauc']:.4f}, rocauc_d: {metrics['rocauc_d']:.4f}")
        # add mean to dataframe
        results.loc[results['modes'] == 'mean', 'AUROC VAE'] = np.mean(aurocs)
        results.loc[results['modes'] == 'mean', 'AUROC Discriminator'] = np.mean(aurocs_discriminator)
        results.loc[results['modes'] == 'mean', 'FPR95 VAE'] = np.mean(fpr95)
        results.loc[results['modes'] == 'mean', 'FPR95 Discriminator'] = np.mean(fpr95_discriminator)
        results.loc[results['modes'] == 'mean', 'Mean Score VAE'] = np.mean(mean_scores)
        results.loc[results['modes'] == 'mean', 'Mean Score Discriminator'] = np.mean(mean_scores_discriminator)
        print(np.mean(in_array), np.mean(in_array_discriminator))

# save results as csv
results.to_csv('xray_results_adv.csv', index=False)