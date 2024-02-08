from models.Diffusion.Diffusion import *
from data.Dataloaders import *
import torch
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Lambda, ToPILImage
import cv2

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader, input_size, channels = pick_dataset('svhn', 'val', 1, normalize=True)
model = DDPM(n_features=input_size, in_channels=channels, channel_scale_factors=(1, 2, 4,)).to(device)
model.load_state_dict(torch.load('./../../models/DDPMsvhn.pth'))
scheduler = LinearScheduler(beta_start=0.0001, beta_end=0.02, timesteps=300)
forward_diffusion_model = ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, reverse_transform=reverse_transform)
clusters = 2
# get one sample from the loader
for batch in loader:
    x = batch[0]
    x = x.to(device)
    noise = torch.randn_like(x)
    t = torch.zeros(1).long().to(device)
    x_noisy = forward_diffusion_model.q_sample(x_start=x, t=t, noise=noise)

    mask = model.generate_masks(x_noisy, t)
    mask = mask[0].cpu().detach().numpy()
    mask = mask.transpose(1, 2, 0)
    mask = mask.reshape(mask.shape[0] * mask.shape[1], mask.shape[2])
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(mask)
    # reshape the labels to the shape 7,7
    labels = kmeans.labels_.reshape(8, 8)
    distances = np.zeros((clusters,32,32), dtype=np.float32)
    
    for j in range(clusters):
        label_mask = np.zeros_like(labels)
        label_mask[labels == j] = 1
        label_mask = torch.tensor(label_mask).unsqueeze(0).unsqueeze(0).to(device)

        t = (torch.ones(1)*85).long().to(device)
        x_noisy = forward_diffusion_model.q_sample(x_start=x, t=t, noise=noise)

        mask_a = model.modulation(x_noisy, t, label_mask, factor = 5)
        mask_b = model.modulation(x_noisy, t, label_mask, factor = -5)
        print(mask_a.shape)
        print(mask_b.shape)
        # get euclidean distance between the two masks
        distance = torch.norm(mask_a - mask_b, dim=1)
        print(distance.shape)
        distance = distance.cpu().detach().numpy()
        plt.imshow(distance[0], cmap='gray')
        plt.show()
        distances[j] = distance

    segmentations = np.argmax(distances, axis=0)
    segmentations_add = np.zeros((32, 32, 3))
    for i in range(clusters):
        segmentations_add[segmentations == i, i] = 1
    # make x a numpy array
    x = x.cpu().detach().numpy()
    x = x * 0.5 + 0.5
    #x = x.reshape(28, 28)
    x = x.squeeze()
    x = x.transpose(1, 2, 0)
    # make it rgb
    #x = np.stack((x, x, x), axis=2)
    x = x*0.9 + segmentations_add*0.1
    # clip the values
    x = np.clip(x, 0, 1)
    # plot the distance
    plt.imshow(x)
    plt.show()