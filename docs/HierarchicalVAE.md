# Hierarchical Variational Autoencoders (Hierarchical VAEs)

NVAE introduced a deep hierarchical VAE designed for image generation through depth-wise separable convolutions and batch normalisation. NVAE has a residual parameterization for normal distributions and uses spectral regularisation to stabilise training.

## Parameters

| Argument                 | Description                             | Default         | Choices                                                                                  |
|--------------------------|-----------------------------------------|-----------------|------------------------------------------------------------------------------------------|
| `--train`                | Train model                             | `False`         |                                                                                          |
| `--sample`               | Sample model                            | `False`         |                                                                                          |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--batch_size`           | Batch size                              | `256`           |                                                                                          |
| `--n_epochs`             | Number of epochs                        | `100`           |                                                                                          |
| `--lr`                   | Learning rate                           | `0.01`          |                                                                                          |
| `--latent_dim`           | Latent dimension                        | `512`           |                                                                                          |
| `--checkpoint`           | Checkpoint path                         | `None`          |                                                                                          |
| `--sample_and_save_freq` | Sample and save frequency               | `5`             |                                                                                          |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python HVAE.py --help

## Training

HVAE can be trained similarly to other models in the Zoo:

    python HVAE.py --train --dataset svhn

## Sampling

For sampling you must provide the HVAE checkpoint:

    python HVAE.py --sample --dataset svhn --checkpoint ./../../models/HierarchicalVAE/HVAE_svhn.pt