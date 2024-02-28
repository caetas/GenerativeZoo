from models.Flow.Glow import *
from data.Dataloaders import *
import torch

train_loader,_,_ = pick_dataset(dataset_name='mnist', batch_size=128, normalize=False, size=32, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Glow(image_shape=(32,32,1), hidden_channels=128, K=8, L=3, actnorm_scale=1.0, flow_permutation = 'invconv', flow_coupling = 'affine', LU_decomposed = False, y_classes=10, learn_top=True, y_condition=False, device=device).to(device)
model.train_model(train_loader)