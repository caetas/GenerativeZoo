from models.SGM.VanillaSGM import train, sample, outlier_detection
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import data_raw_dir, models_dir
import torch
import os


def mnist_train_loader(batch_size):
    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def mnist_val_loader(batch_size):
    validation_data = datasets.MNIST(root=data_raw_dir, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return validation_loader

def fashion_mnist_train_loader(batch_size):
    training_data = datasets.FashionMNIST(root=data_raw_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def fashion_mnist_val_loader(batch_size):
    validation_data = datasets.FashionMNIST(root=data_raw_dir, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return validation_loader

dataloader = mnist_train_loader(64)
val_loader_a = mnist_val_loader(64)
val_loader_b = fashion_mnist_val_loader(64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#train(dataloader, device, n_epochs=50)
#sample(os.path.join(models_dir, 'VanillaSGM.pt'), sampler_type='Euler-Maruyama', device = device, num_samples=16, num_steps=500)
outlier_detection(os.path.join(models_dir, 'VanillaSGM.pt'),val_loader_a, val_loader_b, device)

