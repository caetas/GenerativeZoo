# Vanilla Score-Based Generative Model (Vanilla SGM)

Score-based Generative Models (SGMs) model the diffusion process using a Stochastic Differential Equation (SDE) instead of a Markov chain, as seen in traditional DDPMs. In this approach, the time-dependent score function is employed to construct the reverse-time SDE, which describes the process of removing noise from a sample to return it to the original data distribution. This reverse-time SDE is then solved numerically, often using techniques like Euler-Maruyama or Langevin dynamics, to generate samples from the original data distribution using samples from a simple prior distribution. To facilitate this process, a time-dependent score-based model is trained to approximate the score function. This model learns to estimate the score function at different timesteps, allowing for the construction of the reverse-time SDE and the generation of high-quality samples.

**This model supports `Accelerate` for Multi-GPU and Mixed Precision Training.**

## Parameters

| **Parameter**             | **Description**                               | **Default**   | **Choices**                                                                                                        |
|---------------------------|-----------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------|
| `--dataset`               | Dataset name                                 | `mnist`       | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--batch_size`            | Batch size                                   | `128`         |                                                                                                                    |
| `--n_epochs`              | Number of epochs                             | `100`         |                                                                                                                    |
| `--train`                 | Train model                                  | `False`       |                                                                                                                    |
| `--sample`                | Sample model                                 | `False`       |                                                                                                                    |
| `--model_channels`        | Number of features                           | `64`          |                                                                                                                    |
| `--num_res_blocks`        | Number of residual blocks per downsample     | `2`           |                                                                                                                    |
| `--attention_resolutions` | Downsample rates for attention               | `[4]`         |                                                                                                                    |
| `--dropout`               | Dropout probability                          | `0.0`         |                                                                                                                    |
| `--channel_mult`          | Channel multiplier for UNet levels           | `[1, 2, 2]`   |                                                                                                                    |
| `--conv_resample`         | Use learned convolutions for resampling      | `True`        |                                                                                                                    |
| `--dims`                  | Signal dimensionality (1D, 2D, or 3D)        | `2`           |                                                                                                                    |
| `--num_heads`             | Number of attention heads                    | `4`           |                                                                                                                    |
| `--num_head_channels`     | Fixed channel width per attention head       | `32`          |                                                                                                                    |
| `--use_scale_shift_norm`  | Use FiLM-like conditioning mechanism         | `False`       |                                                                                                                    |
| `--resblock_updown`       | Use residual blocks for up/downsampling      | `False`       |                                                                                                                    |
| `--use_new_attention_order` | Use alternative attention pattern           | `False`       |                                                                                                                    |
| `--solver`                | Solver for ODE                               | `euler`       | `dopri5`, `rk4`, `dopri8`, `euler`, `bosh3`, `adaptive_heun`, `midpoint`, `explicit_adams`, `implicit_adams`        |
| `--outlier_detection`     | Outlier detection                            | `False`       |                                                                                                                    |
| `--atol`                  | Absolute tolerance                           | `1e-6`        |                                                                                                                    |
| `--rtol`                  | Relative tolerance                           | `1e-6`        |                                                                                                                    |
| `--eps`                   | Smallest timestep for numeric stability      | `1e-3`        |                                                                                                                    |
| `--lr`                    | Learning rate                                | `5e-4`        |                                                                                                                    |
| `--sigma`                 | Sigma                                        | `25.0`        |                                                                                                                    |
| `--checkpoint`            | Checkpoint path                              | `None`        |                                                                                                                    |
| `--num_samples`           | Number of samples                            | `16`          |                                                                                                                    |
| `--num_steps`             | Number of steps                              | `100`         |                                                                                                                    |
| `--sample_and_save_freq`  | Sample and save frequency                    | `10`          |                                                                                                                    |
| `--no_wandb`              | Disable wandb logging                        | `False`       |                                                                                                                    |
| `--num_workers`           | Number of workers for Dataloader             | `0`           |                                                                                                                    |
| `--ema_rate`              | Exponential moving average rate              | `0.999`       |                                                                                                                    |
| `--conditional`           | Conditional                                  | `False`       |                                                                                                                    |
| `--warmup`                | Warmup epochs                                | `10`          |                                                                                                                    |
| `--decay`                 | Weight decay rate                            | `0.0`         |                                                                                                                    |
| `--latent`                | Use latent implementation                    | `False`       |                                                                                                                    |
| `--size`                  | Size of the original image                   | `None`        |                                                                                                                    |
| `--n_classes`             | Number of classes                            | `10`          |                                                                                                                    |
| `--cfg`                   | Label guidance                               | `1.0`         |                                                                                                                    |
| `--drop_prob`             | Dropout probability                          | `0.1`         |                                                                                                                    |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VanSGM.py --help

## Training

The Time-dependent score-based model is trained on random timesteps ranging from 0 to 1 to mimic a continuous time interval. The process is somewhat similar to what we have seen in DDPMs, although we don´t explicitly predict the noise that was added.

    accelerate launch VanSGM.py --train --dataset svhn

## Sampling

To sample from SGMs, various numerical solvers can be employed. These solvers include classic methods like Euler-Maruyama and Predictor-Corrector, which are commonly used in stochastic differential equation (SDE) simulations. Additionally, ODE solvers can also be leveraged to make the process deterministic and reduce sampling times, although they sacrifice image quality. All these solvers are avilable via `--sampler_type`.

    python VanSGM.py --sample --dataset svhn --sampler_type PC --checkpoint ./../../models/VanillaSGM/VanSGM_svhn.pt

## Outlier Detection

To detect out-of-distribution samples, we use the loss function. In-distribution samples, resembling the training data, have low loss values, while out-of-distribution samples, unlike the training data, result in higher loss values.

    python VanSGM.py --outlier_detection --dataset svhn --out_dataset cifar10 --checkpoint ./../../models/VanillaSGM/VanSGM_svhn.pt