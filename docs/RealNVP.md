# RealNVP

This work implements the Real-valued Non-Volume Preserving (RealNVP) transformations, a set of powerful invertible and learnable transformations, resulting in an unsupervised learning algorithm with exact log-likelihood computation, exact sampling, exact inference of latent variables, and an interpretable latent space.

## Parameters

| Argument                    | Description                                       | Default     | Choices                                                       |
|-----------------------------|---------------------------------------------------|-------------|---------------------------------------------------------------|
| `--train`                   | Train model                                       | `False`     |                                                               |
| `--sample`                  | Sample model                                      | `False`     |                                                               |
| `--outlier_detection`       | Outlier detection                                 | `False`     |                                                               |
| `--dataset`                 | Dataset name                                      | `'mnist'`   | `'mnist'`, `'cifar10'`, `'fashionmnist'`, `'chestmnist'`, `'octmnist'`, `'tissuemnist'`, `'pneumoniamnist'`, `'svhn'` |
| `--out_dataset`             | Outlier dataset name                              | `'fashionmnist'` | `'mnist'`, `'cifar10'`, `'fashionmnist'`, `'chestmnist'`, `'octmnist'`, `'tissuemnist'`, `'pneumoniamnist'`, `'svhn'` |
| `--batch_size`              | Batch size                                        | `128`       |                                                               |
| `--n_epochs`                | Number of epochs                                  | `100`       |                                                               |
| `--lr`                      | Learning rate                                     | `1e-3`      |                                                               |
| `--weight_decay`            | Weight decay                                      | `1e-5`      |                                                               |
| `--max_grad_norm`           | Max grad norm                                     | `100.0`     |                                                               |
| `--sample_and_save_freq`    | Sample and save frequency                         | `5`         |                                                               |
| `--num_scales`              | Number of scales                                  | `2`         |                                                               |
| `--mid_channels`            | Mid channels                                      | `64`        |                                                               |
| `--num_blocks`              | Number of blocks                                  | `8`         |                                                               |
| `--checkpoint`              | Checkpoint path                                   | `None`      |                                                               |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python RealNVP.py --help

## Training

You can train this model with the following command:

    python RealNVP.py --train --dataset octmnist

## Sampling

To sample, please provide the checkpoint:

    python RealNVP.py --sample --dataset octmnist --checkpoint ./../../models/RealNVP/RealNVP_octmnist.pt

## Outlier Detection

Outlier Detection is performed by using the NLL scores generated by the model:

    python RealNVP.py --outlier_detection --dataset octmnist --out_dataset mnist --checkpoint ./../../models/RealNVP/RealNVP_octmnist.pt