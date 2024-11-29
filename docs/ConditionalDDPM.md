# Conditional Denoising Diffusion Probabilistic Model

The Conditional Denoising Diffusion Probabilistic Model (CDDPM) is akin to the standard DDPM, with the additional incorporation of class embeddings into the training and sampling process. These class embeddings provide the model with information about the specific class or category of the data being generated.

## Parameters

| **Parameter**           | **Description**                          | **Default**        | **Choices**                                                                                                        |
|--------------------------|------------------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `--train`               | Train model                              | `False`            |                                                                                                                    |
| `--sample`              | Sample model                             | `False`            |                                                                                                                    |
| `--batch_size`          | Batch size                               | `128`              |                                                                                                                    |
| `--n_epochs`            | Number of epochs                         | `100`              |                                                                                                                    |
| `--lr`                  | Learning rate                            | `0.001`            |                                                                                                                    |
| `--timesteps`           | Number of timesteps                      | `500`              |                                                                                                                    |
| `--beta_start`          | Beta start                               | `0.0001`           |                                                                                                                    |
| `--beta_end`            | Beta end                                 | `0.02`             |                                                                                                                    |
| `--dataset`             | Dataset name                             | `mnist`            | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--no_wandb`            | Disable Wandb                            | `False`            |                                                                                                                    |
| `--ddpm`                | DDPM                                     | `1.0`              |                                                                                                                    |
| `--checkpoint`          | Checkpoint path                          | `None`             |                                                                                                                    |
| `--sample_timesteps`    | Number of timesteps for sampling          | `500`              |                                                                                                                    |
| `--model_channels`      | Number of features                       | `64`               |                                                                                                                    |
| `--num_res_blocks`      | Number of residual blocks per downsample  | `2`                |                                                                                                                    |
| `--attention_resolutions`| Downsample rates for attention           | `[4]`              |                                                                                                                    |
| `--dropout`             | Dropout probability                      | `0.0`              |                                                                                                                    |
| `--channel_mult`        | Channel multiplier for UNet levels        | `[1, 2, 2]`        |                                                                                                                    |
| `--conv_resample`       | Use learned convolutions for resampling   | `True`             |                                                                                                                    |
| `--dims`                | Signal dimensionality (1D, 2D, 3D)        | `2`                |                                                                                                                    |
| `--num_heads`           | Number of attention heads per layer       | `4`                |                                                                                                                    |
| `--num_head_channels`   | Fixed channel width per attention head    | `32`               |                                                                                                                    |
| `--use_scale_shift_norm`| Use FiLM-like conditioning mechanism      | `False`            |                                                                                                                    |
| `--resblock_updown`     | Use residual blocks for up/downsampling   | `False`            |                                                                                                                    |
| `--use_new_attention_order`| Use an alternative attention pattern    | `False`            |                                                                                                                    |
| `--n_classes`           | Number of classes                        | `10`               |                                                                                                                    |
| `--sample_and_save_freq`| Sample and save frequency                 | `10`               |                                                                                                                    |
| `--drop_prob`           | Dropout probability                      | `0.1`              |                                                                                                                    |
| `--cfg`             | Guidance weight                          | `0.5`              |                                                                                                                    |
| `--num_workers`         | Number of workers for Dataloader          | `0`                |                                                                                                                    |
| `--warmup`                   | `10`                   | Number of warmup epochs                       |                                                                                                              |
| `--decay`                    | `1e-5`                 | Decay rate                                    |                                                                                                              |
| `--latent`                   | `False`                | Use latent version                            |                                                                                                              |   
| `--ema_rate`                 | `0.999`                | Exponential moving average rate               |                                                                                                              |
| `--size`                     | `None`                 | Size of input image                           |                                                                                                              |


You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python CondDDPM.py --help

## Training

The training process is similar to the one described in [`VanillaDDPM.md`](VanillaDDPM.md).

    python CondDDPM.py --train --dataset mnist --n_classes 10

## Sampling

The sampling process is also similar to a typical DDPm, although the class embedding is also provided at each timestep. Conditional DDPMs can also be adjusted to sample in a deterministic manner:

    python CondDDPM.py --sample --dataset mnist --n_classes 10 --checkpoint ./../../models/ConditionalDDPM/CondDDPM_mnist.pt

## Outlier Detection

**To be implemented.**