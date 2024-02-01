#from models.GANs.ConditionalGAN import *
from models.GANs.VanillaGAN import *
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config import data_raw_dir

def mnist_dataloader():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = MNIST(root=data_raw_dir, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    return dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = mnist_dataloader()
#train(dataloader, n_epochs=100, device=device, lr=0.0002, beta1=0.5, beta2=0.999, latent_dim=100, n_classes=10, img_size=32, channels=1, sample_interval=5)
train(dataloader, n_epochs=100, device=device, lr=0.0002, beta1=0.5, beta2=0.999, latent_dim=100, img_size=32, channels=1, sample_interval=5)
