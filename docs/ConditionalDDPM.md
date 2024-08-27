# Conditional Denoising Diffusion Probabilistic Model

The Conditional Denoising Diffusion Probabilistic Model (CDDPM) is akin to the standard DDPM, with the additional incorporation of class embeddings into the training and sampling process. These class embeddings provide the model with information about the specific class or category of the data being generated.

## Parameters

| Parameter              | Description                               | Default | Choices                                                      |
|------------------------|-------------------------------------------|---------|--------------------------------------------------------------|
| `--train`              | train model                               | `False` |                                                              |
| `--sample`             | sample model                              | `False` |                                                              |
| `--outlier_detection`  | outlier detection                         | `False` |                                                              |
| `--batch_size`         | batch size                                | `128`   |                                                              |
| `--n_epochs`           | number of epochs                          | `100`   |                                                              |
| `--lr`                 | learning rate                             | `0.001` |                                                              |
| `--timesteps`          | number of timesteps                       | `500`   |                                                              |
| `--beta_start`         | beta start                                | `0.0001`|                                                              |
| `--beta_end`           | beta end                                  | `0.02`  |                                                              |
| `--dataset`            | Dataset name                              | `mnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`           | Disable Wandb                             | `False` |                                                              |
| `--ddpm`               | ddpm                                      | `1.0`   |                                                              |
| `--checkpoint`         | checkpoint path                           | `None`  |                                                              |
| `--out_dataset`        | outlier dataset name                      | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`|
| `--sample_timesteps`   | number of timesteps for sampling          | `500`   |                                                              |
| `--n_features`         | number of features                        | `128`   |                                                              |
| `--n_classes`          | number of classes                         | `10`    |                                                              |
| `--sample_and_save_freq` | sample and save frequency              | `10`    |                                                              |
| `--drop_prob`          | dropout probability                       | `0.1`   |                                                              |
| `--guide_w`            | guide weight                              | `0.5`   |                                                              |
| `--ws_test`            | guidance weights for test                 | `[0.0, 0.5, 2.0]` |                                                |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python CondDDPM.py --help

## Training

The training process is similar to the one described in [`VanillaDDPM.md`](VanillaDDPM.md).

    python CondDDPM.py --train --dataset mnist --n_classes 10

## Sampling

The sampling process is also similar to a typical DDPm, although the class embedding is also provided at each timestep. Conditional DDPMs can also be adjusted to sample in a deterministic manner:

    python CondDDPM.py --sample --dataset mnist --n_classes 10 --checkpoint ./../../models/ConditionalDDPM/CondDDPM_mnist.pt

## Outlier Detection

**To be implemented.**