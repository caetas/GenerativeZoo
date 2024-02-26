from models.Flow.RealNVP import *
from data.Dataloaders import *
import torch
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, _, _ = pick_dataset(dataset_name='cifar10', batch_size=64, normalize=False, size=32)
model = RealNVP(device=device, in_channels=3, num_blocks = 8)
model.train_model(train_loader)