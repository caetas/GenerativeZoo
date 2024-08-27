# Rectified Flows

## Parameters

| Argument                     | Description                        | Default           | Choices                                                                                                        |
|------------------------------|------------------------------------|-------------------|--------------------------------------------------------------------------------------------------------------|
| `--train`                    | Train model                        | `False`           |                                                                                                              |
| `--sample`                   | Sample model                       | `False`           |                                                                                                              |
| `--outlier_detection`        | Outlier detection                  | `False`           |                                                                                                              |
| `--dataset`                  | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`                 | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--out_dataset`              | Outlier dataset name               | `fashionmnist`  | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `imagenet` |
| `--batch_size`               | Batch size                         | `128`             |                                                                                                              |
| `--n_epochs`                 | Number of epochs                   | `100`             |                                                                                                              |
| `--lr`                       | Learning rate                      | `5e-4`            |                                                                                                              |
| `--patch_size`               | Patch size                         | `2`               |                                                                                                              |
| `--dim`                      | Dimension                          | `64`              |                                                                                                              |
| `--n_layers`                 | Number of layers                   | `6`               |                                                                                                              |
| `--n_heads`                  | Number of heads                    | `4`               |                                                                                                              |
| `--multiple_of`              | Multiple of                        | `256`             |                                                                                                              |
| `--ffn_dim_multiplier`       | FFN dim multiplier                 | `None`            |                                                                                                              |
| `--norm_eps`                 | Norm eps                           | `1e-5`            |                                                                                                              |
| `--class_dropout_prob`       | Class dropout probability          | `0.1`             |                                                                                                              |
| `--sample_and_save_freq`     | Sample and save frequency          | `5`               |                                                                                                              |
| `--num_classes`              | Number of classes                  | `10`              |                                                                                                              |
| `--checkpoint`               | Checkpoint path                    | `None`            |                                                                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python RF.py --help

## Training

You can train this model with the following command:

    python RF.py --train --dataset mnist

## Sampling

To sample, please provide the checkpoint:

    python RF.py --sample --dataset fashionmnist --checkpoint ./../../models/FlowMatching/FM_mnist.pt