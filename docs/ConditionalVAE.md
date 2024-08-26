# Conditional Variational Autoencoder (Conditional VAE)

The Conditional Variational Autoencoder (Conditional VAE) is an extension of the Vanilla VAE that incorporates additional conditional information during the training and generation process, in this case using a class label.

### Parameters

| Parameter       | Description                           | Default | Choices                                            |
|-----------------|---------------------------------------|---------|----------------------------------------------------|
| `--train`       | Train model                           | `False` |                                                    |
| `--sample`      | Sample model                          | `False` |                                                    |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--batch_size`  | Batch size                            | `128`   |                                                    |
| `--n_epochs`    | Number of epochs                      | `100`   |                                                    |
| `--lr`          | Learning rate                         | `0.0002`|                                                    |
| `--latent_dim`  | Latent dimension                      | `128`   |                                                    |
| `--hidden_dims` | Hidden dimensions                     | `None`  |                                                    |
| `--checkpoint`  | Checkpoint path                       | `None`  |                                                    |
| `--num_samples` | Number of samples                     | `16`    |                                                    |
| `--n_classes`   | Number of classes on dataset          | `10`    |                                                    |
| `--sample_and_save_frequency`| sample and save frequency | `5`    |                                                    |
| `--loss_type`             | Type of loss to evaluate reconstruction            | `mse`    |  `mse`, `ssim`             |
| `--kld_weight`            | KL-Divergence weight                               | `1e-4`   |                            |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python CondVAE.py --help

## Training

The training process for the Conditional VAE is similar to the one described in [`VanillaVAE.md`](VanillaVAE.md). Both models aim to maximize the evidence lower bound (ELBO) by minimizing the reconstruction loss and the KL divergence between the estimated latent distribution and the prior distribution. The reconstruction loss measures the difference between the generated output and the original input, while the KL divergence encourages the latent distribution to match the prior distribution.

To train a model on the MNIST dataset, you can run the provided example script:

    python CondVAE.py --train --dataset mnist --n_classes 10

## Sampling

Sampling from the Conditional VAE is similar to the sampling process of a Vanilla VAE, but class information is added.

1. Sample a point from the latent space. This can be done by sampling from a prior distribution, typically a Gaussian distribution with mean 0 and variance 1. Pick a class and represent it in the required embedding format.

2. Pass the sampled point and the embedding through the decoder network to generate a new data point of the given class.

You can sample from the model you trained on MNIST by running:

    python VanVAE.py --sample --dataset mnist --checkpoint ./../../models/ConditionalVAE/CondVAE_mnist.pt
