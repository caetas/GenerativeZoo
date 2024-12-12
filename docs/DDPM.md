# Denoising Diffusion Probabilistic Model (DDPM)

Denoising Diffusion Probabilistic Models (DDPMs) rely on a diffusion process where noise is progressively added to an image, and a series of denoising steps are applied to recover the original image. Unlike traditional autoregressive models, DDPM directly models the diffusion process, enabling efficient and high-quality image generation.

**This model supports `Accelerate` for Multi-GPU and Mixed Precision Training.**

## Parameters

| **Parameter**           | **Description**                              | **Default**        | **Choices**                                                                                                        |
|--------------------------|----------------------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `--train`               | Train model                                  | `False`            |                                                                                                                    |
| `--sample`              | Sample model                                 | `False`            |                                                                                                                    |
| `--outlier_detection`   | Outlier detection                            | `False`            |                                                                                                                    |
| `--fid`                 | Sample for FID                               | `False`            |                                                                                                                    |
| `--batch_size`          | Batch size                                   | `128`              |                                                                                                                    |
| `--n_epochs`            | Number of epochs                             | `100`              |                                                                                                                    |
| `--lr`                  | Learning rate                                | `1e-3`             |                                                                                                                    |
| `--timesteps`           | Number of timesteps                          | `300`              |                                                                                                                    |
| `--model_channels`      | Number of features                           | `64`               |                                                                                                                    |
| `--num_res_blocks`      | Number of residual blocks per downsample      | `2`                |                                                                                                                    |
| `--attention_resolutions`| Downsample rates at which attention occurs   | `[4]`              |                                                                                                                    |
| `--dropout`             | Dropout probability                          | `0.0`              |                                                                                                                    |
| `--channel_mult`        | Channel multiplier for UNet levels           | `[1, 2, 2]`        |                                                                                                                    |
| `--conv_resample`       | Use learned convolutions for resampling       | `True`             |                                                                                                                    |
| `--dims`                | Signal dimensionality (1D, 2D, or 3D)         | `2`                |                                                                                                                    |
| `--num_heads`           | Number of attention heads per layer           | `4`                |                                                                                                                    |
| `--num_head_channels`   | Fixed channel width per attention head        | `32`               |                                                                                                                    |
| `--use_scale_shift_norm`| Use FiLM-like conditioning mechanism          | `False`            |                                                                                                                    |
| `--resblock_updown`     | Use residual blocks for up/downsampling       | `False`            |                                                                                                                    |
| `--use_new_attention_order`| Use alternative attention pattern           | `False`            |                                                                                                                    |
| `--beta_start`          | Beta start                                   | `0.0001`           |                                                                                                                    |
| `--beta_end`            | Beta end                                     | `0.02`             |                                                                                                                    |
| `--sample_and_save_freq`| Sample and save frequency                     | `5`                |                                                                                                                    |
| `--dataset`             | Dataset name                                 | `mnist`            | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--ddpm`                | DDIM sampling is `0.0`, pure DDPM is `1.0`   | `1.0`              |                                                                                                                    |
| `--checkpoint`          | Checkpoint path                              | `None`             |                                                                                                                    |
| `--num_samples`         | Number of samples                            | `16`               |                                                                                                                    |
| `--out_dataset`         | Outlier dataset name                         | `fashionmnist`     | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--loss_type`           | Loss type                                    | `huber`            | `huber`, `l2`, `l1`                                                                                               |
| `--sample_timesteps`    | Number of timesteps for sampling              | `300`              |                                                                                                                    |
| `--no_wandb`            | Disable wandb logging                        | `False`            |                                                                                                                    |
| `--lpips`               | Use LPIPS for OOD                            | `False`            |                                                                                                                    |
| `--num_workers`         | Number of workers for Dataloader             | `0`                |                                                                                                                    |
| `--recon_factor`        | Reconstruction factor                        | `0.5`              |                                                                                                                    |
| `--warmup`              | Number of warmup epochs                      | `10`               |                                                                                                                    |
| `--decay`               | Decay rate                                   | `1e-5`             |                                                                                                                    |
| `--latent`              | Use latent version                           | `False`            |                                                                                                                    |
| `--ema_rate`            | Exponential moving average rate              | `0.999`            |                                                                                                                    |
| `--size`                | Size of input image                          | `None`             |                                                                                                                    |



You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python DDPM.py --help

## Training

During training, a denoiser, typically implemented as a U-Net, is trained to predict the noise added to an image at a given timestep during the forward diffusion process. The quality of this prediction is evaluated using the Mean Squared Error (MSE) between the predicted noise and the actual noise added to the image. By optimizing the denoiser to minimize this MSE loss, DDPM learns to accurately model the diffusion process.

    python DDPM.py --train --dataset cifar10

## Sampling

During the sampling process, we begin with a noisy input image and predict the noise at each timestep. This prediction helps remove the noise, resulting in a progressively clearer image. In typical DDPMs, additional noise is often added after each denoising step, introducing stochasticity to the process. However, it's possible to make this process deterministic by opting not to add extra noise, resulting in what's known as a Denoising Diffusion Implicit Model (DDIM). **You can sample from this model as a DDIM by adjusting `--ddpm`**. Regardless of whether the process is stochastic or deterministic, sampling involves iterating through a fixed number of timesteps to generate high-quality output images.

    python DDPM.py --sample --dataset cifar10 --checkpoint ./../../models/DDPM/DDPM_cifar10.pt

## Outlier Detection

By leveraging the reconstruction error and the perceptual loss between the reconstruction and the input image we can produce an anomaly score to perform OOD detection.

    python DDPM.py --outlier_detection --lpips --dataset cifar10 --out_dataset svhn --checkpoint ./../../models/DDPM/DDPM_cifar10.pt

## FID Sampling

To sample 50K images for FID computation:

    python DDPM.py --fid --num_samples 200 --dataset cifar10 --checkpoint ./../../models/DDPM/DDPM_cifar10.pt