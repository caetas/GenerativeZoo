from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


class Horse2Zebra(Dataset):
    def __init__(self, root, dataset, transform=None, train = True, distribution = 'A'):
        self.root = root
        self.transform = transform
        self.dataset = dataset
        self.train = train
        self.distribution = distribution
        if self.train:
            self.files = sorted(os.listdir(os.path.join(root, dataset, 'train' + distribution)))
        else:
            self.files = sorted(os.listdir(os.path.join(root, dataset, 'test' + distribution)))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        if self.train:
            img = Image.open(os.path.join(self.root, self.dataset, 'train' + self.distribution, self.files[index])).convert('RGB')
        else:
            img = Image.open(os.path.join(self.root, self.dataset, 'test' + self.distribution, self.files[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
def get_horse2zebra_dataloader(root, dataset, batch_size, train = True, distribution = 'A', input_size = 128):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    set = Horse2Zebra(root, dataset, transform, train, distribution)
    dataloader = DataLoader(set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader