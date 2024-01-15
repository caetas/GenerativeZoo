import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import data_raw_dir
from tqdm import trange
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda, ToPILImage, Resize, ToTensor, CenterCrop
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LinearScheduler():
    def __init__(self, beta_start=0.0001, beta_end=0.02, timesteps=1000):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._linear_beta_schedule()
        alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_one_by_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = self._compute_forward_diffusion_alphas(alphas_cumprod)
        self.posterior_variance = self._compute_posterior_variance(alphas_cumprod_prev, alphas_cumprod)

    def _compute_forward_diffusion_alphas(self, alphas_cumprod):
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    
    def _compute_posterior_variance(self, alphas_cumprod_prev, alphas_cumprod):
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        return self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def _linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)


def extract_time_index(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class ForwardDiffusion():
    def __init__(self, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract_time_index(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_time_index(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noisy_image(self, x_start, t, noise=None):
        x_noisy = self.q_sample(x_start, t, noise)
        #noisy_image = self.reverse_transform(x_noisy.squeeze())
        return x_noisy

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

def get_noisy_array(images, t, forward_diffusion):
    out_array = np.zeros_like(images)
    k = 0
    for i in images:
        # reshape i to 1, 28, 28
        i = i.reshape(1, 28, 28)
        # convert to tensor
        i = torch.from_numpy(i)
        # get noisy image
        noisy_image = forward_diffusion.get_noisy_image(x_start=i, t=torch.tensor([t]))
        # convert to numpy
        noisy_image = noisy_image.numpy()
        # flatten the image
        noisy_image = noisy_image.reshape(784)
        # add to out_array
        out_array[k] = noisy_image
        k += 1

    return out_array


dataset = get_mnist_dataset()
# flatten the dataset
dataset = dataset.reshape(60000, 784)
beta_start = 0.0001
beta_end = 0.02
timesteps = 300

t_range = [t for t in range(0, timesteps)]

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

scheduler = LinearScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)
forward_diffusion = ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod)

pca = PCA(n_components=2)
pca.fit(dataset)

noisy_imgs = dataset.copy()

for t in range(0, timesteps, timesteps // 15):

    embeddings = pca.transform(noisy_imgs)
    fig = plt.figure(figsize=(10,10))
    plt.hist2d(embeddings[:,0], embeddings[:,1], bins=(300,300), cmap='jet')
    # remove axis
    plt.axis('off')
    plt.title('t = ' + str(t))
    # a«save image
    plt.savefig('./mnist_pca_' + str(t) + '.png')
    # add t as title
    plt.close(fig)
    noisy_imgs = get_noisy_array(dataset, t, forward_diffusion)

# create gif
import imageio.v2 as imageio
import os
# get all the files in the directory .png
files = os.listdir('./')
files = [file for file in files if '.png' in file]
# sort the files
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# create a list of images
images = []
for file in files:
    images.append(imageio.imread(file))
# save the images as gif
imageio.mimsave('./mnist_pca.gif', images, fps=5)


pca = PCA(n_components=1)
pca.fit(dataset)

noisy_imgs = dataset.copy()

for t in range(0, timesteps, timesteps // 15):

    embeddings = pca.transform(noisy_imgs)
    fig = plt.figure(figsize=(10,10))
    plt.hist(embeddings, bins=300, color='blue')
    # remove axis
    plt.axis('off')
    plt.title('t = ' + str(t))
    # a«save image
    plt.savefig('./mnist_pca_' + str(t) + '.png')
    # add t as title
    plt.close(fig)
    noisy_imgs = get_noisy_array(dataset, t, forward_diffusion)

# create gif
import imageio.v2 as imageio
import os
# get all the files in the directory .png
files = os.listdir('./')
files = [file for file in files if '.png' in file]
# sort the files
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# create a list of images
images = []
for file in files:
    images.append(imageio.imread(file))
# save the images as gif
imageio.mimsave('./single_mnist_pca.gif', images, fps=5)

