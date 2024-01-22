from models.Score.NCSN import train
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import data_raw_dir

def mnist_train_loader(batch_size):
    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize((28,28)),
                                      transforms.ToTensor(),
                                  ]))

    training_loader = DataLoader(training_data, 
                                 batch_size=batch_size, 
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=True)
    return training_loader


dataloader = mnist_train_loader(64)

train(dataloader)