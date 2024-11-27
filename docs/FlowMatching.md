# Flow Matching

## Parameters

| Argument                     | Default           | Help                                          | Choices                                                                                                      |
|------------------------------|-------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `--train`                    | `False`           | Train model                                   |                                                                                                              |
| `--sample`                   | `False`           | Sample model                                  |                                                                                                              |
| `--batch_size`               | `256`             | Batch size                                    |                                                                                                              |
| `--n_epochs`                 | `100`             | Number of epochs                              |                                                                                                              |
| `--lr`                       | `1e-3`            | Learning rate                                 |                                                                                                              |
| `--model_channels`           | `64`              | Number of features                            |                                                                                                              |
| `--num_res_blocks`           | `2`               | Number of residual blocks per downsample      |                                                                                                              |
| `--attention_resolutions`    | `[4]`             | Downsample rates for attention                |                                                                                                              |
| `--dropout`                  | `0.0`             | Dropout probability                           |                                                                                                              |
| `--channel_mult`             | `[1, 2, 2]`       | Channel multiplier for UNet levels           |                                                                                                              |
| `--conv_resample`            | `True`            | Use learned convolutions for resampling       |                                                                                                              |
| `--dims`                     | `2`               | Signal dimensionality (1D, 2D, 3D)            |                                                                                                              |
| `--num_heads`                | `4`               | Number of attention heads per layer           |                                                                                                              |
| `--num_head_channels`        | `32`              | Fixed channel width per attention head        |                                                                                                              |
| `--use_scale_shift_norm`     | `False`           | Use FiLM-like conditioning mechanism          |                                                                                                              |
| `--resblock_updown`          | `False`           | Use residual blocks for up/downsampling       |                                                                                                              |
| `--use_new_attention_order`  | `False`           | Use an alternative attention pattern          |                                                                                                              |
| `--sample_and_save_freq`     | `5`               | Sample and save frequency                     |                                                                                                              |
| `--dataset`                  | `mnist`           | Dataset name                                  | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--checkpoint`               | `None`            | Checkpoint path                               |                                                                                                              |
| `--num_samples`              | `16`              | Number of samples                             |                                                                                                              |
| `--out_dataset`              | `fashionmnist`    | Outlier dataset name                          | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--outlier_detection`        | `False`           | Enable outlier detection                      |                                                                                                              |
| `--interpolation`            | `False`           | Enable interpolation                          |                                                                                                              |
| `--solver_lib`               | `none`            | Solver library                                | `torchdiffeq`, `zuko`, `none`                                                                                |
| `--step_size`                | `0.1`             | Step size for ODE solver                      |                                                                                                              |
| `--solver`                   | `dopri5`          | Solver for ODE                                | `dopri5`, `rk4`, `dopri8`, `euler`, `bosh3`, `adaptive_heun`, `midpoint`, `explicit_adams`, `implicit_adams` |
| `--no_wandb`                 | `False`           | Disable Wandb logging                         |                                                                                                              |
| `--num_workers`              | `0`               | Number of workers for Dataloader              |                                                                                                              |
| `--warmup`                   | `10`              | Number of warmup epochs                       |                                                                                                              |
| `--decay`                    | `1e-5`            | Decay rate                                    |                                                                                                              |



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