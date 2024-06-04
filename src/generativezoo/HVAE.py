from models.VAE.HierarchicalVAE import *
from data.Dataloaders import *

dataloader, img_size, channels = pick_dataset('mnist', size=32, batch_size=256)

model = HierarchicalVAE(512, (img_size, img_size), channels)
model.train_model(dataloader, 10)