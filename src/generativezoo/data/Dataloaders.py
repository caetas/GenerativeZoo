from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from config import data_raw_dir, data_dir
from medmnist import ChestMNIST, TissueMNIST, OCTMNIST, PneumoniaMNIST
import os
from glob import glob
from PIL import Image
import numpy as np
import torch
import tarfile
import io
from tqdm import tqdm
from datasets import load_dataset


def cifar_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    training_data = datasets.CIFAR10(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 32, 3

def cifar_val_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    validation_data = datasets.CIFAR10(root=data_raw_dir, train=False, download=True, transform=transform)
    
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 3
    else:
        return validation_loader, 32, 3

def mnist_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers = num_workers)
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 32, 1

def mnist_val_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    validation_data = datasets.MNIST(root=data_raw_dir, train=False, download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   pin_memory=True)
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 32, 1

def chestmnist_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        training_data = ChestMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)
    else:
        training_data = ChestMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 32, 1

def chestmnist_val_loader(batch_size, normalize = False, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        validation_data = ChestMNIST(root=data_raw_dir, split='val', download=True, transform=transform, size = size)
    else:
        validation_data = ChestMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 32, 1

def octmnist_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        training_data = OCTMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)
    else:
        training_data = OCTMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 32, 1

def octmnist_val_loader(batch_size, normalize = False, input_shape = None):
        
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        validation_data = OCTMNIST(root=data_raw_dir, split='val', download=True, transform=transform, size = size)
    else:
        validation_data = OCTMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 32, 1

def tissuemnist_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        training_data = TissueMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)
    else:
        training_data = TissueMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 32, 1

def tissuemnist_val_loader(batch_size, normalize = False, input_shape = None):
                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        validation_data = TissueMNIST(root=data_raw_dir, split='val', download=True, transform=transform, size = size)
    else:
        validation_data = TissueMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 32, 1

def pneumoniamnist_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
                            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])

    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        training_data = PneumoniaMNIST(root=data_raw_dir, split='train', download=True, transform=transform, size = size)
    else:
        training_data = PneumoniaMNIST(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)

    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 32, 1 

def pneumoniamnist_val_loader(batch_size, normalize = False, input_shape = None):
                                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    if input_shape is not None:
        size = min([64, 128, 224], key=lambda x: abs(x - input_shape))
        validation_data = PneumoniaMNIST(root=data_raw_dir, split='val', download=True, transform=transform, size = size)
    else:
        validation_data = PneumoniaMNIST(root=data_raw_dir, split='val', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 32, 1

def fashion_mnist_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    training_data = datasets.FashionMNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 32, 1

def fashion_mnist_val_loader(batch_size, normalize = False, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    validation_data = datasets.FashionMNIST(root=data_raw_dir, train=False, download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 32, 1


def svhn_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
        
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    training_data = datasets.SVHN(root=data_raw_dir, split='train', download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 32, 3

def svhn_val_loader(batch_size, normalize = False, input_shape = None):
            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(32),
            transforms.ToTensor(),
        ])
    
    validation_data = datasets.SVHN(root=data_raw_dir, split='test', download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 3
    else:
        return validation_loader, 32, 3

class TinyImageNetDataset(Dataset):
    
        def __init__(self, root, train = False, transform=None):
            self.root = root
            self.transform = transform
            self.train = train
            self.imgs = []
            self.label = []
            if train:
                # to get self.imgs iterate over all class folders and all images in each class
                classes = os.listdir(os.path.join(root, 'tiny-imagenet-200', 'train'))
                classes.sort()
                for i in range(200):
                    class_folder = os.path.join(root, 'tiny-imagenet-200', 'train', classes[i], 'images')
                    for img in os.listdir(class_folder):
                        self.imgs.append(os.path.join(class_folder, img))
                        self.label.append(i)
            else:
                self.imgs = glob(os.path.join(root, 'tiny-imagenet-200', 'test', 'images', '*.JPEG'))
                self.label = [0]*len(self.imgs)
            
        def __len__(self):
            return len(self.imgs)
        
        def __getitem__(self, idx):
            img = Image.open(self.imgs[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            # if image name contains good, label is 0, else 1
            return img, self.label[idx]


def tinyimagenet_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
        
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    
    training_data = TinyImageNetDataset(root=data_raw_dir, train = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 64, 3

def tinyimagenet_test_loader(batch_size, normalize = False, input_shape = None):
                
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    
    test_data = TinyImageNetDataset(root=data_raw_dir, train = False, transform=transform)

    test_loader = DataLoader(test_data, 
                            batch_size=batch_size, 
                            shuffle=True,
                            pin_memory=True)
    
    if input_shape is not None:
        return test_loader, input_shape, 3
    else:
        return test_loader, 64, 3
    
def cifar100_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
    
    training_data = datasets.CIFAR100(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 32, 3

def cifar100_val_loader(batch_size, normalize = False, input_shape = None):
                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
    
    validation_data = datasets.CIFAR100(root=data_raw_dir, train=False, download=True, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 3
    else:
        return validation_loader, 32, 3 
    
def places365_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
    
    data_list = os.listdir(data_raw_dir)
    if 'data_256_standard' in data_list:
        download = False
    else:
        download = True

    training_data = datasets.Places365(root=data_raw_dir, split='train-standard', transform=transform, download=download, small=True)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 128, 3
    
def places365_test_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
    
    data_list = os.listdir(data_raw_dir)
    if 'val_256' in data_list:
        download = False
    else:
        download = True

    validation_data = datasets.Places365(root=data_raw_dir, split='val', transform=transform, small=True, download=download)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return validation_loader, input_shape, 3
    else:
        return validation_loader, 128, 3
    
def dtd_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
        
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
    
    training_data = datasets.DTD(root=data_raw_dir, split='train', transform=transform, download=True)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers = num_workers)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 128, 3
    
def dtd_test_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):
                
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
            transforms.ToTensor(),
        ])
    
    test_data = datasets.DTD(root=data_raw_dir, split='test', transform=transform, download=True)

    test_loader = DataLoader(test_data, 
                            batch_size=batch_size, 
                            shuffle=True,
                            pin_memory=True,
                            num_workers = num_workers)
    
    if input_shape is not None:
        return test_loader, input_shape, 3
    else:
        return test_loader, 128, 3

class ImageNetDataset(Dataset):
    def __init__(self, transform_fn, train = False):

        self.dataset = load_dataset('benjamin-paine/imagenet-1k-256x256')
        self.transform_fn = transform_fn
        self.train = train
        self.dataset.set_transform(self.transform_fn)
        self.dataset = self.dataset['train' if train else 'test']

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]['pixel_values'], self.dataset[idx]['label']

def imagenet_train_loader(batch_size, normalize = False, input_shape = None, num_workers = 0):

        if normalize:
            transform = transforms.Compose([
                transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])

        else:
            transform = transforms.Compose([
                transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
                transforms.ToTensor(),
            ])

        # Define a function to apply transformations to each dataset sample
        def transform_fn(examples):
            examples['pixel_values'] = [transform(image) for image in examples['image']]
            return examples

        dataset = ImageNetDataset(transform_fn, train = True)

        training_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers = num_workers)
        
        if input_shape is not None:
            return training_loader, input_shape, 3
        else:
            return training_loader, 128, 3
        
        
def imagenet_val_loader(batch_size, normalize = False, input_shape = None):

        if normalize:
            transform = transforms.Compose([
                transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])

        else:
            transform = transforms.Compose([
                transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)),
                transforms.ToTensor(),
            ])

        # Define a function to apply transformations to each dataset sample
        def transform_fn(examples):
            examples['pixel_values'] = [transform(image) for image in examples['image']]
            return examples
        
        dataset = ImageNetDataset(transform_fn, train = False)

        validation_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers = 0)
        
        if input_shape is not None:
            return validation_loader, input_shape, 3
        else:
            return validation_loader, 128, 3   

def pick_dataset(dataset_name, mode = 'train', batch_size = 64, normalize = False, good = True, size = None, num_workers = 0):
    if dataset_name == 'mnist':
        if mode == 'train':
            return mnist_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return mnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'chestmnist':
        if mode == 'train':
            return chestmnist_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return chestmnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'octmnist':
        if mode == 'train':
            return octmnist_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return octmnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'tissuemnist':
        if mode == 'train':
            return tissuemnist_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return tissuemnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'pneumoniamnist':
        if mode == 'train':
            return pneumoniamnist_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return pneumoniamnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'fashionmnist':
        if mode == 'train':
            return fashion_mnist_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return fashion_mnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'svhn':
        if mode == 'train':
            return svhn_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return svhn_val_loader(batch_size, normalize, size)
    elif dataset_name == 'cifar10':
        if mode == 'train':
            return cifar_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return cifar_val_loader(batch_size, normalize, size)
    elif dataset_name == 'cifar100':
        if mode == 'train':
            return cifar100_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return cifar100_val_loader(batch_size, normalize, size)
    elif dataset_name == 'tinyimagenet':
        if mode == 'train':
            return tinyimagenet_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return tinyimagenet_test_loader(batch_size, normalize, size)
    elif dataset_name == 'places365':
        if mode == 'train':
            return places365_train_loader(batch_size, normalize, size, num_workers = num_workers)
        elif mode == 'val':
            return places365_test_loader(batch_size, normalize, size, num_workers = num_workers)
    elif dataset_name == 'dtd':
        if mode == 'train':
            return dtd_train_loader(batch_size, normalize, size, num_workers = num_workers)
        elif mode == 'val':
            return dtd_test_loader(batch_size, normalize, size, num_workers = num_workers)
    if dataset_name == 'imagenet':
        if mode == 'train':
            return imagenet_train_loader(batch_size, normalize, size, num_workers)
        elif mode == 'val':
            return imagenet_val_loader(batch_size, normalize, size)
    else:
        raise ValueError('Dataset name not found.')