# Flow Matching

## Parameters

| Argument                     | Description                                    | Default           | Choices                                                                                                      |
|------------------------------|------------------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------|
| `--train`                    | Train model                                    | `False`           |                                                                                                              |
| `--sample`                   | Sample model                                   | `False`           |                                                                                                              |
| `--batch_size`               | Batch size                                     | `256`             |                                                                                                              |
| `--n_epochs`                 | Number of epochs                               | `100`             |                                                                                                              |
| `--lr`                       | Learning rate                                  | `1e-3`            |                                                                                                              |
| `--model_channels`           | Number of features                             | `64`              |                                                                                                              |
| `--num_res_blocks`           | Number of residual blocks per downsample       | `2`               |                                                                                                              |
| `--attention_resolutions`    | Downsample rates for attention                 | `[4]`             |                                                                                                              |
| `--dropout`                  | Dropout probability                            | `0.0`             |                                                                                                              |
| `--channel_mult`             | Channel multiplier for UNet levels            | `[1, 2, 2]`       |                                                                                                              |
| `--conv_resample`            | Use learned convolutions for resampling        | `True`            |                                                                                                              |
| `--dims`                     | Signal dimensionality (1D, 2D, 3D)             | `2`               |                                                                                                              |
| `--num_heads`                | Number of attention heads per layer            | `4`               |                                                                                                              |
| `--num_head_channels`        | Fixed channel width per attention head         | `32`              |                                                                                                              |
| `--use_scale_shift_norm`     | Use FiLM-like conditioning mechanism           | `False`           |                                                                                                              |
| `--resblock_updown`          | Use residual blocks for up/downsampling        | `False`           |                                                                                                              |
| `--use_new_attention_order`  | Use an alternative attention pattern           | `False`           |                                                                                                              |
| `--sample_and_save_freq`     | Sample and save frequency                      | `5`               |                                                                                                              |
| `--dataset`                  | Dataset name                                   | `mnist`           | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--checkpoint`               | Checkpoint path                                | `None`            |                                                                                                              |
| `--num_samples`              | Number of samples                              | `16`              |                                                                                                              |
| `--out_dataset`              | Outlier dataset name                           | `fashionmnist`    | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--outlier_detection`        | Enable outlier detection                       | `False`           |                                                                                                              |
| `--interpolation`            | Enable interpolation                           | `False`           |                                                                                                              |
| `--solver_lib`               | Solver library                                 | `none`            | `torchdiffeq`, `zuko`, `none`                                                                                |
| `--step_size`                | Step size for ODE solver                       | `0.1`             |                                                                                                              |
| `--solver`                   | Solver for ODE                                 | `dopri5`          | `dopri5`, `rk4`, `dopri8`, `euler`, `bosh3`, `adaptive_heun`, `midpoint`, `explicit_adams`, `implicit_adams` |
| `--no_wandb`                 | Disable Wandb logging                          | `False`           |                                                                                                              |
| `--num_workers`              | Number of workers for Dataloader               | `0`               |                                                                                                              |
| `--warmup`                   | Number of warmup epochs                        | `10`              |                                                                                                              |
| `--decay`                    | Decay rate                                     | `1e-5`            |                                                                                                              |


You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python FM.py --help

## Training

You can train this model with the following command:

    python FM.py --train --dataset mnist

## Sampling

To sample, please provide the checkpoint:

    python FM.py --sample --dataset mnist --checkpoint ./../../models/FlowMatching/FM_mnist.pt

## Outlier Detection

Outlier Detection is performed by using the NLL scores generated by the model:

    python FM.py --outlier_detection --dataset mnist --out_dataset fashionmnist --checkpoint ./../../models/FlowMatching/FM_mnist.pt