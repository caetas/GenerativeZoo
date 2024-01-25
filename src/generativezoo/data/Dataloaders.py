from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import data_raw_dir
from medmnist import ChestMNIST, TissueMNIST, OCTMNIST, PneumoniaMNIST

def cifar_train_loader(batch_size, normalize = False):

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    training_data = datasets.CIFAR10(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def cifar_val_loader(batch_size, normalize = False):

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    validation_data = datasets.CIFAR10(root=data_raw_dir, train=False, download=True, transform=transform)
    
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return validation_loader

def mnist_train_loader(batch_size, normalize = False):

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def mnist_val_loader(batch_size, normalize = False):

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    validation_data = datasets.MNIST(root=data_raw_dir, train=False, download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    return validation_loader

def chestmnist_train_loader(batch_size, normalize = False):

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    training_data = ChestMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def chestmnist_val_loader(batch_size, normalize = False):
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    validation_data = ChestMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def octmnist_train_loader(batch_size, normalize = False):
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    training_data = OCTMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def octmnist_val_loader(batch_size, normalize = False):
        
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    validation_data = OCTMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def tissuemnist_train_loader(batch_size, normalize = False):
            
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    training_data = TissueMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def tissuemnist_val_loader(batch_size, normalize = False):
                    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    validation_data = TissueMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def pneumoniamnist_train_loader(batch_size, normalize = False):
                            
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    training_data = PneumoniaMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader  

def pneumoniamnist_val_loader(batch_size, normalize = False):
                                    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    validation_data = PneumoniaMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def fashion_mnist_train_loader(batch_size, normalize = False):
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    training_data = datasets.FashionMNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def fashion_mnist_val_loader(batch_size, normalize = False):
    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    validation_data = datasets.FashionMNIST(root=data_raw_dir, train=False, download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader


def svhn_train_loader(batch_size, normalize = False):
        
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    training_data = datasets.SVHN(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def svhn_val_loader(batch_size, normalize = False):
            
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    validation_data = datasets.SVHN(root=data_raw_dir, split='test', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def pick_dataset(dataset_name, mode = 'train', batch_size = 64, normalize = False):
    if dataset_name == 'mnist':
        if mode == 'train':
            return mnist_train_loader(batch_size, normalize), 28, 1
        elif mode == 'val':
            return mnist_val_loader(batch_size, normalize), 28, 1
    elif dataset_name == 'chestmnist':
        if mode == 'train':
            return chestmnist_train_loader(batch_size, normalize), 28, 1
        elif mode == 'val':
            return chestmnist_val_loader(batch_size, normalize), 28, 1
    elif dataset_name == 'octmnist':
        if mode == 'train':
            return octmnist_train_loader(batch_size, normalize), 28, 1
        elif mode == 'val':
            return octmnist_val_loader(batch_size, normalize), 28, 1
    elif dataset_name == 'tissuemnist':
        if mode == 'train':
            return tissuemnist_train_loader(batch_size, normalize), 28, 1
        elif mode == 'val':
            return tissuemnist_val_loader(batch_size, normalize), 28, 1
    elif dataset_name == 'pneumoniamnist':
        if mode == 'train':
            return pneumoniamnist_train_loader(batch_size, normalize), 28, 1
        elif mode == 'val':
            return pneumoniamnist_val_loader(batch_size, normalize), 28, 1
    elif dataset_name == 'fashionmnist':
        if mode == 'train':
            return fashion_mnist_train_loader(batch_size, normalize), 28, 1
        elif mode == 'val':
            return fashion_mnist_val_loader(batch_size, normalize), 28, 1
    elif dataset_name == 'svhn':
        if mode == 'train':
            return svhn_train_loader(batch_size, normalize), 32, 3
        elif mode == 'val':
            return svhn_val_loader(batch_size, normalize), 32, 3
    elif dataset_name == 'cifar10':
        if mode == 'train':
            return cifar_train_loader(batch_size, normalize), 32, 3
        elif mode == 'val':
            return cifar_val_loader(batch_size, normalize), 32, 3
    else:
        raise ValueError('Dataset name not found.')