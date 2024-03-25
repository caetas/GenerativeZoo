from config import data_raw_dir
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from models.GAN.PrescribedGAN import *
from utils.util import parse_args_PresGAN
from data.Dataloaders import *
import pandas as pd

class MNISTC(Dataset):
    def __init__(self, root, transform=None, corruption='impulse_noise'):
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.array = np.load(os.path.join(data_raw_dir, 'mnist_c', f'{corruption}', 'test_images.npy'))

    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, idx):
        img = self.array[idx].squeeze()
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, 0
    
def create_corrupted_loader(corruption, batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    dataset = MNISTC(root=data_raw_dir, transform=transform, corruption=corruption)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

args = parse_args_PresGAN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_loader, input_size, channels = pick_dataset(dataset_name='mnist',batch_size=args.batch_size, size=32, normalize=True, mode = "val")
corruption_types = os.listdir(os.path.join(data_raw_dir, 'mnist_c'))
corruption_types.sort() 
print(corruption_types)

# dataframe to store results
results = pd.DataFrame(columns=['corruption_type', 'AUROC', 'FPR95', 'Mean Scores'], index=range(len(corruption_types) + 1))

# pre fill the dataframe
counter = 0
for corruption in corruption_types:
    results.loc[counter] = ({'corruption_type': f'{corruption}', 'AUROC': 0.0, 'FPR95': 0.0, 'Mean Scores': 0.0})
    counter += 1
# append mean
results.loc[counter] = ({'corruption_type': 'Mean', 'AUROC': 0.0, 'FPR95': 0.0, 'Mean Scores': 0.0})

args.discriminator_checkpoint = "./../../models/PrescribedGAN/PresDisc_mnist_100.pt"
args.sigma_checkpoint = "./../../models/PrescribedGAN/PresSigma_mnist_100.pt"
args.nz = 100

model = PresGAN(imgSize=input_size, nz=args.nz, ngf = args.ngf, ndf = args.ndf, nc = channels, device = device, beta1 = args.beta1, lrD = args.lrD, lrG = args.lrG, n_epochs = args.n_epochs, sigma_lr=args.sigma_lr,
                    num_gen_images=args.num_gen_images, restrict_sigma=args.restrict_sigma, sigma_min=args.sigma_min, sigma_max=args.sigma_max, stepsize_num=args.stepsize_num, lambda_=args.lambda_,
                    burn_in=args.burn_in, num_samples_posterior=args.num_samples_posterior, leapfrog_steps=args.leapfrog_steps, hmc_learning_rate=args.hmc_learning_rate, hmc_opt_accept=args.hmc_opt_accept, flag_adapt=args.flag_adapt, 
                    sample_and_save_freq=args.sample_and_save_freq, dataset=args.dataset)
model.load_checkpoints(generator_checkpoint=args.checkpoint, discriminator_checkpoint=args.discriminator_checkpoint, sigma_checkpoint=args.sigma_checkpoint)

auroc_list = []
fpr95_list = []
scores_list = []
in_array = None
for corruption in corruption_types:
    out_loader = create_corrupted_loader(corruption, args.batch_size)
    auroc, fpr95, in_array, scores = model.outlier_detection(in_loader, out_loader, in_array=in_array, display=False)
    auroc_list.append(auroc)
    fpr95_list.append(fpr95)
    scores_list.append(scores)
    results.loc[results['corruption_type'] == corruption, 'AUROC'] = auroc
    results.loc[results['corruption_type'] == corruption, 'FPR95'] = fpr95
    results.loc[results['corruption_type'] == corruption, 'Mean Scores'] = scores
    print(f"Corruption: {corruption}, AUROC: {auroc:.4f}")

# print mean
auroc_list = np.array(auroc_list)
print(np.mean(auroc_list))
print(np.mean(in_array))
results.loc[results['corruption_type'] == 'Mean', 'AUROC'] = np.mean(auroc_list)
results.loc[results['corruption_type'] == 'Mean', 'FPR95'] = np.mean(fpr95_list)
results.loc[results['corruption_type'] == 'Mean', 'Mean Scores'] = np.mean(scores_list)

# save results
results.to_csv(f'mnist_c_presgan.csv', index=False)

