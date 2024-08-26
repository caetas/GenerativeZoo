# Flow++

Flow++ is a generative model that aims to learn the underlying probability distribution of a given dataset. This work improves upon three limiting design choices employed by flow-based models in prior work: the use of uniform noise for dequantization, the use of inexpressive affine flows, and the use of purely convolutional conditioning networks in coupling layers.

## Parameters

| Argument                  | Description                              | Default         | Choices                                                                                  |
|---------------------------|------------------------------------------|-----------------|------------------------------------------------------------------------------------------|
| `--train`                 | Train model                              | `False`         |                                                                                          |
| `--sample`                | Sample from model                        | `False`         |                                                                                          |
| `--outlier_detection`     | Outlier detection                        | `False`         |                                                                                          |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--out_dataset`           | Outlier dataset name                     | `fashionmnist`| `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`  |
| `--batch_size`            | Batch size                               | `8`             |                                                                                          |
| `--n_epochs`              | Number of epochs                         | `100`           |                                                                                          |
| `--lr`                    | Learning rate                            | `1e-3`          |                                                                                          |
| `--warm_up`               | Warm up                                  | `200`           |                                                                                          |
| `--grad_clip`             | Gradient clip                            | `1.0`           |                                                                                          |
| `--drop_prob`             | Dropout probability                      | `0.2`           |                                                                                          |
| `--num_blocks`            | Number of blocks                         | `10`            |                                                                                          |
| `--num_components`        | Number of components in the mixture      | `32`            |                                                                                          |
| `--num_dequant_blocks`    | Number of blocks in dequantization       | `2`             |                                                                                          |
| `--num_channels`          | Number of channels in Flow++             | `96`            |                                                                                          |
| `--use_attn`              | Use attention                            | `False`         |                                                                                          |
| `--sample_and_save_freq`  | Sample interval                          | `5`             |                                                                                          |
| `--checkpoint`            | Checkpoint path to VQVAE                 | `None`          |                                                                                          |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python FlowPP.py --help

## Training

You can train this model with the following command:

    python FlowPP.py --train --dataset mnist

## Sampling

To sample, please provide the checkpoint:

    python FlowPP.py --sample --dataset fashionmnist --checkpoint ./../../models/FlowPP/FlowPP_mnist.pt

## Outlier Detection

Outlier Detection is performed by using the NLL scores generated by the model:

    python FlowPP.py --outlier_detection --dataset mnist --out_dataset fashionmnist --checkpoint ./../../models/FlowPP/FlowPP_mnist.pt