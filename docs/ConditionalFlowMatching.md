# Conditional Flow Matching

**This model supports `Accelerate` for Multi-GPU and Mixed Precision Training.**

## Parameters

| **Argument**                     | **Description**                                    | **Default**        | **Choices**                                                                                                        |
|-----------------------------------|----------------------------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `--train`                         | Train model                                        | `False`            |                                                                                                                    |
| `--sample`                        | Sample model                                       | `False`            |                                                                                                                    |
| `--batch_size`                    | Batch size                                         | `256`              |                                                                                                                    |
| `--n_epochs`                      | Number of epochs                                   | `100`              |                                                                                                                    |
| `--lr`                            | Learning rate                                      | `1e-3`             |                                                                                                                    |
| `--model_channels`                | Number of features                                 | `64`               |                                                                                                                    |
| `--num_res_blocks`                | Number of residual blocks per downsample           | `2`                |                                                                                                                    |
| `--attention_resolutions`         | Downsample rates for attention                     | `[4]`              |                                                                                                                    |
| `--dropout`                       | Dropout probability                                | `0.0`              |                                                                                                                    |
| `--channel_mult`                  | Channel multiplier for UNet levels                 | `[1, 2, 2]`        |                                                                                                                    |
| `--conv_resample`                 | Use learned convolutions for resampling            | `True`             |                                                                                                                    |
| `--dims`                          | Signal dimensionality (1D, 2D, 3D)                 | `2`                |                                                                                                                    |
| `--num_heads`                     | Number of attention heads per layer                | `4`                |                                                                                                                    |
| `--num_head_channels`             | Fixed channel width per attention head             | `32`               |                                                                                                                    |
| `--use_scale_shift_norm`          | Use FiLM-like conditioning mechanism               | `False`            |                                                                                                                    |
| `--resblock_updown`               | Use residual blocks for up/downsampling            | `False`            |                                                                                                                    |
| `--use_new_attention_order`       | Use an alternative attention pattern               | `False`            |                                                                                                                    |
| `--sample_and_save_freq`          | Sample and save frequency                          | `5`                |                                                                                                                    |
| `--dataset`                       | Dataset name                                       | `mnist`            | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--no_wandb`                      | Disable Wandb                                      | `False`            |                                                                                                                    |
| `--checkpoint`                    | Checkpoint path                                    | `None`             |                                                                                                                    |
| `--num_samples`                   | Number of samples                                  | `16`               |                                                                                                                    |
| `--solver_lib`                    | Solver library                                     | `none`             | `torchdiffeq`, `zuko`, `none`                                                                                      |
| `--step_size`                     | Step size for ODE solver                           | `0.1`              |                                                                                                                    |
| `--solver`                        | Solver for ODE                                     | `dopri5`           | `dopri5`, `rk4`, `dopri8`, `euler`, `bosh3`, `adaptive_heun`, `midpoint`, `explicit_adams`, `implicit_adams`        |
| `--n_classes`                     | Number of classes                                  | `10`               |                                                                                                                    |
| `--dropout_prob`                  | Probability of dropping conditioning during training| `0.2`              |                                                                                                                    |
| `--cfg`                           | Guidance scale                                     | `2.0`              |                                                                                                                    |
| `--num_workers`                   | Number of workers for Dataloader                   | `0`                |                                                                                                                    |
| `--warmup`                        | Number of warmup epochs                            | `10`               |                                                                                                                    |
| `--decay`                         | Weight decay of learning rate                      | `0`                |                                                                                                                    |
| `--latent`                        | Use latent version                                 | `False`            |                                                                                                                    |
| `--ema_rate`                      | Exponential moving average rate                    | `0.999`            |                                                                                                                    |
| `--size`                          | Size of input image                                | `None`             |                                                                                                                    |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python CondFM.py --help

## Training

You can train this model with the following command:

    accelerate launch CondFM.py --train --dataset mnist

## Sampling

To sample, please provide the checkpoint:

    python CondFM.py --sample --dataset mnist --checkpoint ./../../models/FlowMatching/FM_mnist.pt

## Outlier Detection

Outlier Detection is performed by using the NLL scores generated by the model:

    python CondFM.py --outlier_detection --dataset mnist --out_dataset fashionmnist --checkpoint ./../../models/FlowMatching/FM_mnist.pt