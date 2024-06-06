# Hierarchical Variational Autoencoders (Hierarchical VAEs)

NVAE introduced a deep hierarchical VAE designed for image generation through depth-wise separable convolutions and batch normalisation. NVAE has a residual parameterization for normal distributions and uses spectral regularisation to stabilise training.

## Parameters

| Argument                 | Description                             | Default         | Choices                                                                                  |
|--------------------------|-----------------------------------------|-----------------|------------------------------------------------------------------------------------------|
| `--train`                | Train model                             | `False`         |                                                                                          |
| `--sample`               | Sample model                            | `False`         |                                                                                          |
| `--dataset`              | Dataset name                            | `'mnist'`       | `'mnist'`, `'cifar10'`, `'fashionmnist'`, `'chestmnist'`, `'octmnist'`, `'tissuemnist'`, `'pneumoniamnist'`, `'svhn'`  |
| `--batch_size`           | Batch size                              | `256`           |                                                                                          |
| `--n_epochs`             | Number of epochs                        | `100`           |                                                                                          |
| `--lr`                   | Learning rate                           | `0.01`          |                                                                                          |
| `--latent_dim`           | Latent dimension                        | `512`           |                                                                                          |
| `--checkpoint`           | Checkpoint path                         | `None`          |                                                                                          |
| `--sample_and_save_freq` | Sample and save frequency               | `5`             |                                                                                          |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python HVAE.py --help

## Training

The PresGAN can be trained in a similar fashion to other GANs in the zoo:

    python HVAE.py --train --dataset svhn

## Sampling

For sampling you must provide the generator checkpoint:

    python HVAE.py --sample --dataset svhn --checkpoint ./../../models/HierarchicalVAE/HVAE_svhn.pt