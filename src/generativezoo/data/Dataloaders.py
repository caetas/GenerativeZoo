from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from config import data_raw_dir
from medmnist import ChestMNIST, TissueMNIST, OCTMNIST, PneumoniaMNIST
import os
from glob import glob
from PIL import Image

def cifar_train_loader(batch_size, normalize = False, input_shape = None):

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
                                 pin_memory=True)
    
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

def mnist_train_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
        ])

    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True)
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 28, 1

def mnist_val_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
        return validation_loader, 28, 1

def chestmnist_train_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
                                 pin_memory=True)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 28, 1

def chestmnist_val_loader(batch_size, normalize = False, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
        return validation_loader, 28, 1

def octmnist_train_loader(batch_size, normalize = False, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
                                pin_memory=True)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 28, 1

def octmnist_val_loader(batch_size, normalize = False, input_shape = None):
        
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
        return validation_loader, 28, 1

def tissuemnist_train_loader(batch_size, normalize = False, input_shape = None):
            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
                                pin_memory=True)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 28, 1

def tissuemnist_val_loader(batch_size, normalize = False, input_shape = None):
                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
        return validation_loader, 28, 1

def pneumoniamnist_train_loader(batch_size, normalize = False, input_shape = None):
                            
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
                                pin_memory=True)

    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 28, 1 

def pneumoniamnist_val_loader(batch_size, normalize = False, input_shape = None):
                                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
        return validation_loader, 28, 1

def fashion_mnist_train_loader(batch_size, normalize = False, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
        ])
    
    training_data = datasets.FashionMNIST(root=data_raw_dir, train=True, download=True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 28, 1

def fashion_mnist_val_loader(batch_size, normalize = False, input_shape = None):
    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_shape) if input_shape is not None else transforms.Resize(28),
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
        return validation_loader, 28, 1


def svhn_train_loader(batch_size, normalize = False, input_shape = None):
        
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
                                pin_memory=True)
    
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
    
def mvtec_toothbrush_train_loader(batch_size, normalize = False, input_shape = None):
            
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
    
    training_data = MVTecDataset(root=data_raw_dir, dataset = 'toothbrush', train = True, good = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 128, 3

def mvtec_toothbrush_val_loader(batch_size, normalize = False, good = True, input_shape = None):
                
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
    
    validation_data = MVTecDataset(root=data_raw_dir, dataset = 'toothbrush', train = False, good = good, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 3
    else:
        return validation_loader, 128, 3

def mvtec_bottle_train_loader(batch_size, normalize = False, input_shape = None):
        
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
    
    training_data = MVTecDataset(root=data_raw_dir, dataset = 'bottle', train = True, good = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
   
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 128, 3
    
def mvtec_bottle_val_loader(batch_size, normalize = False, good = True, input_shape = None):
                    
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
    
    validation_data = MVTecDataset(root=data_raw_dir, dataset = 'bottle', train = False, good = good, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 3
    else:
        return validation_loader, 128, 3

def textile_train_loader(batch_size, normalize = False, input_shape = None):
        
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
    
    training_data = MVTecDataset(root=data_raw_dir, dataset = 'TextileDefects32', train = True, good = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 32, 1

def textile_val_loader(batch_size, normalize = False, good = True, input_shape = None):
                    
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
    
    validation_data = MVTecDataset(root=data_raw_dir, dataset = 'TextileDefects32', train = False, good = good, transform=transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 32, 1

def headct_train_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    
    training_data = MVTecDatasetFull(root=data_raw_dir, dataset = 'headct', train = True, transform=transform)

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
   
    if input_shape is not None:
        return training_loader, input_shape, 1
    else:
        return training_loader, 64, 1

def headct_val_loader(batch_size, normalize = False, input_shape = None):

    if normalize:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    
    validation_data = MVTecDatasetFull(root=data_raw_dir, dataset = 'headct', train = False, transform = transform)

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 1
    else:
        return validation_loader, 64, 1
    

def cityscapes_train_loader(batch_size, normalize = False, input_shape = None):

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
    
    training_data = datasets.Cityscapes(root=data_raw_dir, split='train', transform=transform, target_transform=transforms.Compose([transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)), transforms.ToTensor()]))

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return training_loader, input_shape, 3
    else:
        return training_loader, 128, 3
    
def cityscapes_val_loader(batch_size, normalize = False, input_shape = None):

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
    
    validation_data = datasets.Cityscapes(root=data_raw_dir, split='val', transform=transform, target_transform=transforms.Compose([transforms.Resize((input_shape,input_shape)) if input_shape is not None else transforms.Resize((128,128)), transforms.ToTensor()]))

    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)
    
    if input_shape is not None:
        return validation_loader, input_shape, 3
    else:
        return validation_loader, 128, 3

def pick_dataset(dataset_name, mode = 'train', batch_size = 64, normalize = False, good = True, size = None):
    if dataset_name == 'mnist':
        if mode == 'train':
            return mnist_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return mnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'chestmnist':
        if mode == 'train':
            return chestmnist_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return chestmnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'octmnist':
        if mode == 'train':
            return octmnist_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return octmnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'tissuemnist':
        if mode == 'train':
            return tissuemnist_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return tissuemnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'pneumoniamnist':
        if mode == 'train':
            return pneumoniamnist_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return pneumoniamnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'fashionmnist':
        if mode == 'train':
            return fashion_mnist_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return fashion_mnist_val_loader(batch_size, normalize, size)
    elif dataset_name == 'svhn':
        if mode == 'train':
            return svhn_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return svhn_val_loader(batch_size, normalize, size)
    elif dataset_name == 'cifar10':
        if mode == 'train':
            return cifar_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return cifar_val_loader(batch_size, normalize, size)
    elif dataset_name == 'bottle':
        if mode == 'train':
            return mvtec_bottle_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return mvtec_bottle_val_loader(batch_size, normalize, good, size)
    elif dataset_name == 'toothbrush':
        if mode == 'train':
            return mvtec_toothbrush_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return mvtec_toothbrush_val_loader(batch_size, normalize, good, size)
    elif dataset_name == 'textile':
        if mode == 'train':
            return textile_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return textile_val_loader(batch_size, normalize, good, size)
    elif dataset_name == 'headct':
        if mode == 'train':
            return headct_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return headct_val_loader(batch_size, normalize, size)
    elif dataset_name == 'cityscapes':
        if mode == 'train':
            return cityscapes_train_loader(batch_size, normalize, size)
        elif mode == 'val':
            return cityscapes_val_loader(batch_size, normalize, size)
    else:
        raise ValueError('Dataset name not found.')