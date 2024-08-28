# Glow

Glow is a simple type of generative flow using an invertible 1x1 convolution. Although it is a generative model optimized towards the plain log-likelihood objective, it is capable of efficient realistic-looking synthesis and manipulation of large images.

## Parameters

| Argument             | Description                           | Default | Choices                                              |
|----------------------|---------------------------------------|---------|------------------------------------------------------|
| `--train`            | Train model                           | `False` |                                                      |
| `--sample`           | Sample from model                     | `False` |                                                      |
| `--outlier_detection`| Outlier detection                     | `False` |                                                      |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--out_dataset`      | Outlier dataset name                  | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--batch_size`       | Batch size                            | `128`   |                                                      |
| `--n_epochs`         | Number of epochs                      | `100`   |                                                      |
| `--lr`               | Learning rate                         | `0.0002`|                                                      |
| `--hidden_channels`  | Hidden channels                       | `64`    |                                                      |
| `--K`                | Number of layers per block            | `8`     |                                                      |
| `--L`                | Number of blocks                      | `3`     |                                                      |
| `--actnorm_scale`    | Act norm scale                        | `1.0`   |                                                      |
| `--flow_permutation` | Flow permutation                      |`invconv`| `invconv`, `shuffle`, `reverse`                      |
| `--flow_coupling`    | Flow coupling                         |`affine` | `additive`, `affine`                                 |
| `--LU_decomposed`    | Train with LU decomposed 1x1 convs    |`False`  |                                                      |
| `--learn_top`        | Learn top layer (prior)               | `False` |                                                      |
| `--y_condition`      | Class Conditioned Glow                | `False` |                                                      |
| `--y_weight`         | Weight of class condition             | `0.01`  |                                                      |
| `--num_classes`      | Number of classes                     | `10`    |                                                      |
| `--sample_and_save_freq` | Sample and save frequency         | `5`     |                                                      |
| `--checkpoint`       | Checkpoint path                       | `None`  |                                                      |
| `--n_bits`           | Number of bits                        | `8`     |                                                      |
| `--max_grad_clip`    | Max Grad clip                         | `0.0`   |                                                      |
| `--max_grad_norm`    | Max Grad Norm                         | `0.0`   |                                                      |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |
| `--warmup`   | Number of warmup epochs   | `10`     |                                                              |
| `--decay`   | weight decay of learning rate   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python GLOW.py --help

## Training

You can train this model with the following command:

    python GLOW.py --train --dataset octmnist

## Sampling

To sample, please provide the checkpoint:

    python GLOW.py --sample --dataset octmnist --checkpoint ./../../models/Glow/Glow_octmnist.pt

## Outlier Detection

Outlier Detection is performed by using the NLL scores generated by the model:

    python GLOW.py --outlier_detection --dataset octmnist --out_dataset mnist --checkpoint ./../../models/Glow/Glow_octmnist.pt
