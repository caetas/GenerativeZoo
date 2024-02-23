from models.Flow.RealNVP import *
from data.Dataloaders import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, _, _ = pick_dataset(dataset_name='mnist', batch_size=8, normalize=False, size=32)
model = RealNVP(device=device, in_channels=1, num_blocks = 8)
model.train_model(train_loader)