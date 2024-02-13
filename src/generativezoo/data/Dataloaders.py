from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from config import data_raw_dir
from medmnist import ChestMNIST, TissueMNIST, OCTMNIST, PneumoniaMNIST
import os
from glob import glob
from PIL import Image

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

def mnist_train_loader(batch_size, normalize = False, input_shape = 28):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
        ])

    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    return training_loader

def mnist_val_loader(batch_size, normalize = False, input_shape = 28):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape),
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

def pneumoniamnist_train_loader(batch_size, normalize = False, size = 28):
                            
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    training_data = PneumoniaMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader  

def pneumoniamnist_val_loader(batch_size, normalize = False, size = 28):
                                    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    validation_data = PneumoniaMNIST(root=data_raw_dir, split='val', download=True, transform=transform, size = size)

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

class MVTecDataset(Dataset):

    def __init__(self, root, dataset = 'bottle', train = False, good = True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.good = good
        if train:
            self.imgs = glob(os.path.join(root, dataset, 'train', 'good', '*.png'))
        else:
            if good:
                self.imgs = glob(os.path.join(root, dataset, 'test', 'good', '*.png'))
            else:
                self.imgs = glob(os.path.join(root, dataset, 'test', '*', '*.png'))
                self.imgs = [img for img in self.imgs if 'good' not in img]
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        if self.transform:
            img = self.transform(img)
        return img, 1

class MVTecDatasetFull(Dataset):

    def __init__(self, root, dataset = 'bottle', train = False, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        if train:
            self.imgs = glob(os.path.join(root, dataset, 'train', '*', '*.png'))
        else:
            self.imgs = glob(os.path.join(root, dataset, 'test', '*', '*.png'))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        # if image name contains good, label is 0, else 1
        if 'good' in self.imgs[idx]:
            return img, 0
        else:
            return img, 1
    
def mvtec_toothbrush_train_loader(batch_size, normalize = False):
            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
    
    training_data = MVTecDataset(root=data_raw_dir, dataset = 'toothbrush', train = True, good = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def mvtec_toothbrush_val_loader(batch_size, normalize = False, good = True):
                
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
    
    validation_data = MVTecDataset(root=data_raw_dir, dataset = 'toothbrush', train = False, good = good, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def mvtec_bottle_train_loader(batch_size, normalize = False):
        
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    training_data = MVTecDataset(root=data_raw_dir, dataset = 'bottle', train = True, good = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def mvtec_bottle_val_loader(batch_size, normalize = False, good = True):
                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    validation_data = MVTecDataset(root=data_raw_dir, dataset = 'bottle', train = False, good = good, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def textile_train_loader(batch_size, normalize = False):
        
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    training_data = MVTecDataset(root=data_raw_dir, dataset = 'TextileDefects32', train = True, good = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def textile_val_loader(batch_size, normalize = False, good = True):
                    
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    validation_data = MVTecDataset(root=data_raw_dir, dataset = 'TextileDefects32', train = False, good = good, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def headct_train_loader(batch_size, normalize = False):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    
    training_data = MVTecDatasetFull(root=data_raw_dir, dataset = 'headct', train = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    return training_loader

def headct_val_loader(batch_size, normalize = False):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    
    validation_data = MVTecDatasetFull(root=data_raw_dir, dataset = 'headct', train = False, transform = transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    return validation_loader

def pick_dataset(dataset_name, mode = 'train', batch_size = 64, normalize = False, good = True):
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
    elif dataset_name == 'bottle':
        if mode == 'train':
            return mvtec_bottle_train_loader(batch_size, normalize), 128, 3
        elif mode == 'val':
            return mvtec_bottle_val_loader(batch_size, normalize, good), 128, 3
    elif dataset_name == 'toothbrush':
        if mode == 'train':
            return mvtec_toothbrush_train_loader(batch_size, normalize), 128, 3
        elif mode == 'val':
            return mvtec_toothbrush_val_loader(batch_size, normalize, good), 128, 3
    elif dataset_name == 'textile':
        if mode == 'train':
            return textile_train_loader(batch_size, normalize), 32, 1
        elif mode == 'val':
            return textile_val_loader(batch_size, normalize, good), 32, 1
    elif dataset_name == 'headct':
        if mode == 'train':
            return headct_train_loader(batch_size, normalize), 64, 1
        elif mode == 'val':
            return headct_val_loader(batch_size, normalize), 64, 1
    else:
        raise ValueError('Dataset name not found.')