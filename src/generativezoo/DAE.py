from models.Diffusion.MONAI_DiffAE import DiffAE
import torch
from data.Dataloaders import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffAE(embedding_dimension=512, in_channels=1, num_epochs=1000, lr = 1e-3, inference_timesteps=50)

train_loader, size, in_channels = pick_dataset('headct', 'train', 16, True)
val_loader, _, _ = pick_dataset('headct', 'val', 20, True)

model.train_model(train_loader, val_loader)