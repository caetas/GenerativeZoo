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
from matplotlib import pyplot as plt

class CIFAR10C(Dataset):
    def __init__(self, root, transform=None, corruption='gaussian_noise', severity=5):
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.array = np.load(os.path.join(data_raw_dir, 'cifar10c', f'{corruption}.npy'))[10000*(severity-1):10000*(severity)]
        self.labels = np.load(os.path.join(data_raw_dir, 'cifar10c', 'labels.npy'))[10000*(severity-1):10000*(severity)]

    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, idx):
        img = self.array[idx]
        label = self.labels[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label
    
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

for discriminator_checkpoint in discriminator_list:
    for loss in loss_type:
        print(f"discriminator: {discriminator_checkpoint}, loss: {loss}")
        model = Discriminator(channels=3, d=d).to(device)
        model.load_state_dict(torch.load(os.path.join(checkpoints_base,discriminator_checkpoint)))
        cifar_loader = cifar_test_loader(batch_size)
        auroc_list = []
        fpr95_list = []
        class_list = []
        pbar = tqdm(corruption_types)
        in_array = None
        for c in pbar:
            for i in range(1,6):
                
                # add postfix to corruption type
                corrupted_loader = create_corrupted_loader(c, i, batch_size)
                auroc, fpr95, in_array, scores, class_scores, mean_in_array = model.outlier_detection(cifar_loader, corrupted_loader, in_array=in_array, display=False, device=device)
                # add to dataframe
                if mean_in_array is not None:
                    class_in_scores = np.mean(mean_in_array)

                auroc_list.append(auroc)
                fpr95_list.append(fpr95)
                class_list.append(class_scores)
                results.loc[results['corruption_type'] == f'{c}_{i}', 'AUROC'] = auroc
                results.loc[results['corruption_type'] == f'{c}_{i}', 'FPR95'] = fpr95
                results.loc[results['corruption_type'] == f'{c}_{i}', 'Mean Scores'] = scores

                pbar.set_description(f"corruption: {c}, severity: {i}, auroc: {auroc:.4f}")
        # add mean to dataframe

        results.loc[results['corruption_type'] == 'mean', 'AUROC'] = np.mean(auroc_list)
        results.loc[results['corruption_type'] == 'mean', 'FPR95'] = np.mean(fpr95_list)
        results.loc[results['corruption_type'] == 'mean', 'Mean Scores'] = np.mean(in_array)
        class_list = np.array(class_list)
        class_list = np.mean(class_list, axis=0)

        # plot the class scores as bars
        plt.bar(range(10), class_list-class_in_scores)
        plt.xticks(range(10))
        plt.xlabel('Class')
        plt.ylabel('Score Difference to ID')
        plt.title('Mean Class Difference to ID Scores')
        # add a dotted line with np.mean(in_array)
        #plt.axhline(y=np.mean(in_array), color='r', linestyle='--')
        plt.show()

print(f'Mean AUROC: {np.mean(auroc_list):.4f}')        
# save results as csv
#results.to_csv('cifar10c_results_gan.csv', index=False)