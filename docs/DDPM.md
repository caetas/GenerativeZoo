# Denoising Diffusion Probabilistic Model (DDPM)

Denoising Diffusion Probabilistic Models (DDPMs) rely on a diffusion process where noise is progressively added to an image, and a series of denoising steps are applied to recover the original image. Unlike traditional autoregressive models, DDPM directly models the diffusion process, enabling efficient and high-quality image generation.

## Parameters

| **Argument**               | **Default**            | **Help**                                                                                       | **Choices**                                                                                                                                 |
|----------------------------|-------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `--train`                 | `False`                | Train model                                                                                   |                                                                                                                                          |
| `--sample`                | `False`                | Sample model                                                                                  |                                                                                                                                          |
| `--outlier_detection`     | `False`                | Outlier detection                                                                             |                                                                                                                                          |
| `--batch_size`            | `128`                  | Batch size                                                                                    |                                                                                                                                          |
| `--n_epochs`              | `100`                  | Number of epochs                                                                              |                                                                                                                                          |
| `--lr`                    | `1e-3`                 | Learning rate                                                                                 |                                                                                                                                          |
| `--timesteps`             | `300`                  | Number of timesteps                                                                           |                                                                                                                                          |
| `--model_channels`        | `64`                   | Number of features                                                                            |                                                                                                                                          |
| `--num_res_blocks`        | `2`                    | Number of residual blocks per downsample                                                     |                                                                                                                                          |
| `--attention_resolutions` | `[4]`                  | Downsample rates at which attention will take place                                          |                                                                                                                                          |
| `--dropout`               | `0.0`                  | Dropout probability                                                                           |                                                                                                                                          |
| `--channel_mult`          | `[1, 2, 2]`            | Channel multiplier for each level of the UNet                                                |                                                                                                                                          |
| `--conv_resample`         | `True`                 | Use learned convolutions for upsampling and downsampling                                     |                                                                                                                                          |
| `--dims`                  | `2`                    | Determines if the signal is 1D, 2D, or 3D                                                    |                                                                                                                                          |
| `--num_heads`             | `4`                    | Number of attention heads in each attention layer                                            |                                                                                                                                          |
| `--num_head_channels`     | `32`                   | Use a fixed channel width per attention head                                                 |                                                                                                                                          |
| `--use_scale_shift_norm`  | `False`                | Use a FiLM-like conditioning mechanism                                                       |                                                                                                                                          |
| `--resblock_updown`       | `False`                | Use residual blocks for up/downsampling                                                      |                                                                                                                                          |
| `--use_new_attention_order`| `False`                | Use a different attention pattern for potentially increased efficiency                       |                                                                                                                                          |
| `--beta_start`            | `0.0001`               | Beta start                                                                                   |                                                                                                                                          |
| `--beta_end`              | `0.02`                 | Beta end                                                                                     |                                                                                                                                          |
| `--sample_and_save_freq`  | `5`                    | Sample and save frequency                                                                    |                                                                                                                                          |
| `--dataset`               | `mnist`                | Dataset name                                                                                 | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--ddpm`                  | `1.0`                  | DDIM sampling is 0.0, pure DDPM is 1.0                                                       |                                                                                                                                          |
| `--checkpoint`            | `None`                 | Checkpoint path                                                                              |                                                                                                                                          |
| `--num_samples`           | `16`                   | Number of samples                                                                            |                                                                                                                                          |
| `--out_dataset`           | `fashionmnist`         | Outlier dataset name                                                                         | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--loss_type`             | `huber`                | Loss type                                                                                    | `huber`, `l2`, `l1`                                                                                                                     |
| `--sample_timesteps`      | `300`                  | Number of timesteps                                                                          |                                                                                                                                          |
| `--no_wandb`              | `False`                | Disable wandb logging                                                                        |                                                                                                                                          |
| `--num_workers`           | `0`                    | Number of workers for dataloader                                                            |                                                                                                                                          |
| `--recon_factor`          | `0.5`                  | Reconstruction factor                                                                       |                                                                                                                                          |


You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VanDDPM.py --help

## Training

During training, a denoiser, typically implemented as a U-Net, is trained to predict the noise added to an image at a given timestep during the forward diffusion process. The quality of this prediction is evaluated using the Mean Squared Error (MSE) between the predicted noise and the actual noise added to the image. By optimizing the denoiser to minimize this MSE loss, DDPM learns to accurately model the diffusion process.

    python VanDDPM.py --train --dataset cifar10

## Sampling

During the sampling process, we begin with a noisy input image and predict the noise at each timestep. This prediction helps remove the noise, resulting in a progressively clearer image. In typical DDPMs, additional noise is often added after each denoising step, introducing stochasticity to the process. However, it's possible to make this process deterministic by opting not to add extra noise, resulting in what's known as a Denoising Diffusion Implicit Model (DDIM). **You can sample from this model as a DDIM by adjusting `--ddpm`**. Regardless of whether the process is stochastic or deterministic, sampling involves iterating through a fixed number of timesteps to generate high-quality output images.

    python VanDDPM.py --sample --dataset cifar10 --checkpoint ./../../models/DDPM/VanDDPM_cifar10.pt

## Outlier Detection

By leveraging the noise prediction error at timestep t=0, we can estimate whether a sample is in-distribution or out-of-distribution. In this context, a "good" sample is one that closely matches the training data distribution. Consequently, such a sample should have a lower error in the noise prediction at timestep t=0 compared to out-of-distribution samples.

    python VanDDPM.py --sample --dataset cifar10 --out_dataset svhn --checkpoint ./../../models/DDPM/VanDDPM_cifar10.pt