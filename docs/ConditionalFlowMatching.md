# Conditional Flow Matching

## Parameters

| Argument                     | Description                        | Default           | Choices                                                                                                        |
|------------------------------|------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------|
| `--train`                    | Train model                        | `False`           |                                                                                                              |
| `--sample`                   | Sample model                       | `False`           |                                                                                                              |
| `--batch_size`               | Batch size                         | `256`             |                                                                                                              |
| `--n_epochs`                 | Number of epochs                   | `100`             |                                                                                                              |
| `--lr`                       | Learning rate                      | `1e-3`            |                                                                                                              |
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
| `--sample_and_save_freq`     | Sample and save frequency          | `5`               |                                                                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--checkpoint`               | Checkpoint path                    | `None`            |                                                                                                              |
| `--num_samples`              | Number of samples                  | `16`              |                                                                                                              |
| `--solver_lib`               | Solver library                     | `none`          | `torchdiffeq`, `zuko`, `none`                                                                          |
| `--step_size`                | Step size for ODE solver           | `0.1`             |                                                                                                              |
| `--solver`                   | Solver for ODE                     | `dopri5`        | `dopri5`, `rk4`, `dopri8`, `euler`, `bosh3`, `adaptive_heun`, `midpoint`, `explicit_adams`, `implicit_adams` |
| `--n_classes`              | Number of classes                  | `10`              |                                                                                                              |
| `--dropout_prob`                     | Probability of droping conditioning during training | `0.2`   |                                                                                                              |
| `--cfg`           | Guidance scale                     | `2.0`             |                                                                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |
| `--warmup`   | Number of warmup epochs   | `10`     |                                                              |
| `--decay`   | weight decay of learning rate   | `0`     |                                                              |


You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python CondFM.py --help

## Training

You can train this model with the following command:

    python CondFM.py --train --dataset mnist

## Sampling

To sample, please provide the checkpoint:

    python CondFM.py --sample --dataset mnist --checkpoint ./../../models/FlowMatching/FM_mnist.pt

## Outlier Detection

Outlier Detection is performed by using the NLL scores generated by the model:

    python CondFM.py --outlier_detection --dataset mnist --out_dataset fashionmnist --checkpoint ./../../models/FlowMatching/FM_mnist.pt