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

latent_dim = 1024
d = 64
#hidden_dims_list = [[64, 128, 256]]
batch_size = 256
lr = 0.0002
checkpoints_base = './../../models/VanillaGAN'
discriminator_list = ['VanDisc_cifar10.pt']
loss_type = ['mse']
corruption_types = os.listdir(os.path.join(data_raw_dir, 'cifar10c'))
# remove the one saying labels
corruption_types.remove('labels.npy')
#remove all the .npy
corruption_types = [i.replace('.npy', '') for i in corruption_types]

# create a dataframe to store the results with corruption_types*5 rows
results = pd.DataFrame(columns=['corruption_type', discriminator_list[0]], index=range(len(corruption_types)*5 + 1))

# pre-fill the dataframe with the corruption types and all other columns as 0.0 without using append
counter = 0
for c in corruption_types:
    for i in range(1,6):
        results.loc[counter] = ({'corruption_type': c + '_' + str(i), discriminator_list[0]: 0.0})
        counter += 1
results.loc[counter] = ({'corruption_type': 'mean', discriminator_list[0]: 0.0})

for discriminator_checkpoint in discriminator_list:
    for loss in loss_type:
        print(f"discriminator: {discriminator_checkpoint}, loss: {loss}")
        model = Discriminator(channels=3, d=d).to(device)
        model.load_state_dict(torch.load(os.path.join(checkpoints_base,discriminator_checkpoint)))
        cifar_loader = cifar_test_loader(batch_size)
        aurocs = []
        pbar = tqdm(corruption_types)
        in_array = None
        for c in pbar:
            for i in range(1,6):
                
                # add postfix to corruption type
                corrupted_loader = create_corrupted_loader(c, i, batch_size)
                auroc, in_array = model.outlier_detection(cifar_loader, corrupted_loader, in_array=in_array, display=False, device=device)
                # add to dataframe

                results.loc[results['corruption_type'] == c + '_' + str(i), discriminator_checkpoint] = auroc
                aurocs.append(auroc)

                pbar.set_description(f"corruption: {c}, severity: {i}, auroc: {auroc:.4f}")
        # add mean to dataframe

        results.loc[results['corruption_type'] == 'mean', discriminator_checkpoint] = np.mean(aurocs)
        
# save results as csv
results.to_csv('cifar10c_results_gan.csv', index=False)