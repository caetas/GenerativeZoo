# Variational Autoencoder (Vanilla VAE)

The Vanilla VAE (Variational Autoencoder) is a generative model that learns to encode and decode data. It is commonly used for unsupervised learning tasks such as dimensionality reduction and data generation.

### Parameters

| Parameter       | Description                           | Default | Choices                                            |
|-----------------|---------------------------------------|---------|----------------------------------------------------|
| `--train`       | Train model                           | `False` |                                                    |
| `--sample`      | Sample model                          | `False` |                                                    |
| `--outlier_detection` | Out-of-distribution detection   | `False` |                                                    |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--out_dataset` | Outlier dataset name                  |`mnist`| `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--batch_size`  | Batch size                            | `128`   |                                                    |
| `--n_epochs`    | Number of epochs                      | `100`   |                                                    |
| `--lr`          | Learning rate                         | `0.0002`|                                                    |
| `--latent_dim`  | Latent dimension                      | `128`   |                                                    |
| `--hidden_dims` | Hidden dimensions                     | `None`  |                                                    |
| `--checkpoint`  | Checkpoint path                       | `None`  |                                                    |
| `--num_samples` | Number of samples                     | `16`    |                                                    |
| `--sample_and_save_frequency`| sample and save frequency| `5`     |                                                    |
| `--loss_type`             | Type of loss to evaluate reconstruction            | `mse`    |  `mse`, `ssim`             |
| `--kld_weight`            | KL-Divergence weight                               | `1e-4`   |                            |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VanVAE.py --help

## Training

The Vanilla VAE is trained using a combination of a reconstruction loss and a regularization loss. The reconstruction loss measures the difference between the original data point and its reconstruction. The regularization loss encourages the latent space distribution to follow a prior distribution, typically a Gaussian distribution.

During training, the model learns to minimize the combined loss by adjusting the parameters of the encoder and decoder networks using techniques such as backpropagation and gradient descent.

To train a model on the FashionMNIST dataset, you can run the provided example script:

    python VanVAE.py --train --dataset fashionmnist

## Sampling

Sampling from the Vanilla VAE model allows us to generate new data points based on the learned representations. To sample from the model, we can follow these steps:

1. Sample a point from the latent space. This can be done by sampling from a prior distribution, typically a Gaussian distribution with mean 0 and variance 1.

2. Pass the sampled point through the decoder network to generate a new data point.

You can sample from the model you trained on FashionMNIST by running:

    python VanVAE.py --sample --dataset fashionmnist --checkpoint ./../../models/VanillaVAE/VanVAE_fashionmnist.pt

## Outlier Detection

To detect out-of-distribution samples, we use the loss function as a way to produce an anomaly score. An in-distribution sample should have a low anomaly score, i.e., should be properly reconstructed and its latent space should approximate the prior. On the other hand, an out-of-distribution sample should have a high loss because it is poorly reconstructed and the encoded features do not follow a normal distribution.

    python VanVAE.py --outlier_detection --dataset fashionmnist --out_dataset mnist --checkpoint ./../../models/VanillaVAE/VanVAE_fashionmnist.pt
