# Conditional Flow Matching

## Parameters

| Argument                     | Description                        | Default           | Choices                                                                                                        |
|------------------------------|------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------|
| `--train`                    | Train model                        | `False`           |                                                                                                              |
| `--sample`                   | Sample model                       | `False`           |                                                                                                              |
| `--batch_size`               | Batch size                         | `256`             |                                                                                                              |
| `--n_epochs`                 | Number of epochs                   | `100`             |                                                                                                              |
| `--lr`                       | Learning rate                      | `1e-3`            |                                                                                                              |
| `--n_features`               | Number of features                 | `64`              |                                                                                                              |
| `--init_channels`            | Initial channels                   | `32`              |                                                                                                              |
| `--channel_scale_factors`    | Channel scale factors              | `[1, 2, 2]`       |                                                                                                              |
| `--resnet_block_groups`      | Resnet block groups                | `8`               |                                                                                                              |
| `--use_convnext`             | Use convnext (default: True)       | `True`            |                                                                                                              |
| `--convnext_scale_factor`    | Convnext scale factor (default: 2) | `2`               |                                                                                                              |
| `--sample_and_save_freq`     | Sample and save frequency          | `5`               |                                                                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--checkpoint`               | Checkpoint path                    | `None`            |                                                                                                              |
| `--num_samples`              | Number of samples                  | `16`              |                                                                                                              |
| `--outlier_detection`        | Outlier detection                  | `False`           |                                                                                                              |
| `--interpolation`            | Interpolation                      | `False`           |                                                                                                              |
| `--solver_lib`               | Solver library                     | `none`          | `torchdiffeq`, `zuko`, `none`                                                                          |
| `--step_size`                | Step size for ODE solver           | `0.1`             |                                                                                                              |
| `--solver`                   | Solver for ODE                     | `dopri5`        | `dopri5`, `rk4`, `dopri8`, `euler`, `bosh3`, `adaptive_heun`, `midpoint`, `explicit_adams`, `implicit_adams` |
| `--num_classes`              | Number of classes                  | `10`              |                                                                                                              |
| `--prob`                     | Probability of conditioning during training | `0.5`   |                                                                                                              |
| `--guidance_scale`           | Guidance scale                     | `2.0`             |                                                                                                              |
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