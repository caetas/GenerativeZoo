from models.Diffusion.ConditionalDiffusion import inference
from config import data_raw_dir, models_dir
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


# get entire mnist dataset as a numpy array
def get_mnist_dataset():
    training_data = datasets.MNIST(root=data_raw_dir, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))

    training_loader = DataLoader(training_data, 
                                 batch_size=60000, 
                                 shuffle=True,
                                 pin_memory=True)
    for batch_idx, (data, target) in enumerate(training_loader):
        return data.numpy()


batch_size = 10
n_T = 400
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
guide_w_list = [0.0, 0.5, 2.0]
dataset = get_mnist_dataset()
dataset = dataset.reshape(60000, 784)
pca = PCA(n_components=1)
pca.fit(dataset)

checkpoint_dir = os.path.join(models_dir, 'model_39.pth')

for guide_w in guide_w_list:
    gen_images = inference(checkpoint_dir,'./', batch_size, n_T=n_T, device=device, guide_w=guide_w)

    exit()

    maps = np.zeros((gen_images.shape[1],gen_images.shape[2], gen_images.shape[0])) # n_classes, samp_per_class, n_T
    for i in range(gen_images.shape[1]):
        for j in range(gen_images.shape[0]):
            path = gen_images[j,i,:,:].reshape(gen_images.shape[2], 784)
            embeddings = pca.transform(path)
            maps[i,:,j] = embeddings[:,0]

    for class_type in range(maps.shape[0]):
        plt.figure(figsize=(30,10))
        for i in range(maps.shape[0]):
            if i == class_type:
                continue
            for j in range(maps.shape[1]):
                plt.plot(np.flip(maps[i,j,:]), color='blue', alpha=0.3, linewidth=0.4)
        for j in range(maps.shape[1]):
            plt.plot(np.flip(maps[class_type,j,:]), color='red', linewidth=1)
        plt.savefig('{}_class_{}.png'.format(guide_w,class_type))
        plt.close()
    
    embeddings_zero  = maps[:,:,0].flatten()
    embeddings_end = maps[:,:,-1].flatten()
    # plot histogtam of embeddings in separated images
    plt.figure(figsize=(10,10))
    plt.hist(embeddings_zero, bins=100)
    plt.savefig('{}_t0.png'.format(guide_w))
    plt.close()
    plt.figure(figsize=(10,10))
    plt.hist(embeddings_end, bins=100)
    plt.savefig('{}_t1.png'.format(guide_w))
    plt.close()

