from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import data_raw_dir

def cifar_train_loader(batch_size):
    training_data = datasets.CIFAR10(root=data_raw_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                  ]))

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def cifar_val_loader(batch_size):
    validation_data = datasets.CIFAR10(root=data_raw_dir, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                  ]))

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return validation_loader

def mnist_train_loader(batch_size):
    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
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
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return validation_loader