# Wasserstein GAN with Gradient Penalty (WGAN-GP)

WGAN-GP presents an alternative to clipping weights in typical WGANs: penalize the norm of gradient of the critic with respect to its input. This method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures.

## Parameters

| Argument                  | Description                            | Default  | Choices                                                                 |
|---------------------------|----------------------------------------|----------|-------------------------------------------------------------------------|
| `--train`                 | Train model                            | `False`  |                                                                         |
| `--sample`                | Sample from model                      | `False`  |                                                                         |
| `--outlier_detection`     | Outlier detection                      | `False`  |                                                                         |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--out_dataset`           | Outlier dataset name                   | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--batch_size`            | Batch size                             | `256`    |                                                                         |
| `--n_epochs`              | Number of epochs                       | `100`    |                                                                         |
| `--latent_dim`            | Latent dimension                       | `100`    |                                                                         |
| `--d`                     | D                                      | `64`     |                                                                         |
| `--lrg`                   | Learning rate generator                | `0.0002` |                                                                         |
| `--lrd`                   | Learning rate discriminator            | `0.0002` |                                                                         |
| `--beta1`                 | Beta1                                  | `0.5`    |                                                                         |
| `--beta2`                 | Beta2                                  | `0.999`  |                                                                         |
| `--sample_and_save_freq`  | Sample interval                        | `5`      |                                                                         |
| `--checkpoint`            | Checkpoint path                        | `None`   |                                                                         |
| `--discriminator_checkpoint` | Discriminator checkpoint path        | `None`   |                                                                         |
| `--gp_weight`             | Gradient penalty weight                | `10.0`   |                                                                         |
| `--n_critic`              | Number of critic updates per generator update | `5`  |                                                                         |
| `--n_samples`             | Number of samples                      | `9`      |                                                                         |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python WGAN.py --help

## Training

The training command is similar to the one found in other GANs present in the zoo:

    python WGAN.py --train --dataset cifar10 --latent_dim 1024

## Sampling

Sampling is also close to what is done in other adversarial networks:

    pythow WGAn.py --sample --dataset cifar10 --latent_dim 1024 --checkpoint ./../../models/WassersteinGAN/WGAN_cifar10.py
