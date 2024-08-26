# Vanilla Denoising Diffusion Probabilistic Model (Vanilla DDPM)

Denoising Diffusion Probabilistic Models (DDPMs) rely on a diffusion process where noise is progressively added to an image, and a series of denoising steps are applied to recover the original image. Unlike traditional autoregressive models, DDPM directly models the diffusion process, enabling efficient and high-quality image generation.

## Parameters

| Parameter                   | Description                                     | Default | Choices                                                      |
|-----------------------------|-------------------------------------------------|---------|--------------------------------------------------------------|
| `--train`                   | train model                                     | `False` |                                                              |
| `--sample`                  | sample model                                    | `False` |                                                              |
| `--outlier_detection`       | outlier detection                               | `False` |                                                              |
| `--batch_size`              | batch size                                      | `128`   |                                                              |
| `--n_epochs`                | number of epochs                                | `100`   |                                                              |
| `--lr`                      | learning rate                                   | `0.001` |                                                              |
| `--timesteps`               | number of timesteps                             | `300`   |                                                              |
| `--n_features`              | number of features                              | `64`    |                                                              |
| `--init_channels`           | initial channels                                | `32`    |                                                              |
| `--channel_scale_factors`   | channel scale factors                           | `[1, 2, 2]` |                                                          |
| `--resnet_block_groups`     | resnet block groups                             | `8`     |                                                              |
| `--use_convnext`            | use convnext                                    | `True`  |                                                              |
| `--convnext_scale_factor`   | convnext scale factor                           | `2`     |                                                              |
| `--beta_start`              | beta start                                      | `0.0001`|                                                              |
| `--beta_end`                | beta end                                        | `0.02`  |                                                              |
| `--sample_and_save_freq`    | sample and save frequency                       | `5`     |                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--ddpm`                    | 1.0 is a ddpm, 0.0 is a ddim                                            | `1.0`   |                                                              |
| `--checkpoint`              | checkpoint path                                 | `None`  |                                                              |
| `--num_samples`             | number of samples                               | `16`    |                                                              |
| `--out_dataset`             | outlier dataset name                            | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--loss_type`               | loss type                                       | `huber` | `huber`, `l2`, `l1`                                  |
| `--sample_timesteps`        | number of timesteps for sampling                | `300`   |                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VanDDPM.py --help

## Training

During training, a denoiser, typically implemented as a U-Net, is trained to predict the noise added to an image at a given timestep during the forward diffusion process. The quality of this prediction is evaluated using the Mean Squared Error (MSE) between the predicted noise and the actual noise added to the image. By optimizing the denoiser to minimize this MSE loss, DDPM learns to accurately model the diffusion process.

    python VanDDPM.py --train --dataset cifar10

## Sampling

During the sampling process, we begin with a noisy input image and predict the noise at each timestep. This prediction helps remove the noise, resulting in a progressively clearer image. In typical DDPMs, additional noise is often added after each denoising step, introducing stochasticity to the process. However, it's possible to make this process deterministic by opting not to add extra noise, resulting in what's known as a Denoising Diffusion Implicit Model (DDIM). **You can sample from this model as a DDIM by adjusting `--ddpm`**. Regardless of whether the process is stochastic or deterministic, sampling involves iterating through a fixed number of timesteps to generate high-quality output images.

    python VanDDPM.py --sample --dataset cifar10 --checkpoint ./../../models/VanillaDDPM/VanDDPM_cifar10.pt

## Outlier Detection

By leveraging the noise prediction error at timestep t=0, we can estimate whether a sample is in-distribution or out-of-distribution. In this context, a "good" sample is one that closely matches the training data distribution. Consequently, such a sample should have a lower error in the noise prediction at timestep t=0 compared to out-of-distribution samples.

    python VanDDPM.py --sample --dataset cifar10 --out_dataset svhn --checkpoint ./../../models/VanillaDDPM/VanDDPM_cifar10.pt