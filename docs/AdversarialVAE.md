# Adversarial Variational Autoencoder (Adversarial VAE)

The Adversarial Variational Autoencoder (Adversarial VAE) is a generative model that combines the power of Variational Autoencoders (VAEs) with adversarial training. VAEs are a type of deep generative model that can learn to generate new data samples by capturing the underlying distribution of the training data. Adversarial training, on the other hand, involves training a discriminator network to distinguish between real and generated samples, while simultaneously training the generator network to fool the discriminator. By combining these two techniques, the Adversarial VAE can generate high-quality samples that are both diverse and realistic.

## Parameters

| Argument                  | Description                                        | Default  | Choices                                                                                                                                                                      |
|---------------------------|----------------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--train`                 | Train model                                        | `False`  |                                                                                                                                                                              |
| `--test`                  | Test model                                         | `False`  |                                                                                                                                                                              |
| `--sample`                | Sample model                                       | `False`  |                                                                                                                                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--batch_size`            | Batch size                                         | `128`    |                                                                                                                                                                              |
| `--n_epochs`              | Number of epochs                                   | `100`    |                                                                                                                                                                              |
| `--lr`                    | Learning rate                                      | `0.0002` |                                                                                                                                                                              |
| `--latent_dim`            | Latent dimension                                   | `128`    |                                                                                                                                                                              |
| `--hidden_dims`           | Hidden dimensions                                  | `None`   |                                                                                                                                                                              |
| `--checkpoint`            | Checkpoint path                                    | `None`   |                                                                                                                                                                              |
| `--num_samples`           | Number of samples                                  | `16`     |                                                                                                                                                                              |
| `--gen_weight`            | Generator weight                                   | `0.002`  |                                                                                                                                                                              |
| `--recon_weight`          | Reconstruction weight                              | `0.002`  |                                                                                                                                                                              |
| `--sample_and_save_frequency` | Sample and save frequency                      | `5`      |                                                                                                                                                                              |
| `--outlier_detection`     | Outlier detection                                  | `False`  |                                                                                                                                                                              |
| `--discriminator_checkpoint` | Discriminator checkpoint path                   | `None`   |                                                                                                                                                                              |
| `--out_dataset`           | Outlier dataset name                               | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`      |
| `--loss_type`             | Type of loss to evaluate reconstruction            | `mse`    |  `mse`, `ssim`                                                                                                                                                               |
| `--kld_weight`            | KL-Divergence weight                               | `1e-4`   |                                                                                                                                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |
| `--size`          | Size of image (None uses default for each dataset) | `None` | |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python AdvVAE.py --help

## Training

To train the Generator (the VAE), we do not simply try to minimize the reconstruction loss and the KL divergence. In addition to this, we incorporate two adversarial loss factors, one related with the ability to fool the discriminator with the VAE's reconstructions (adjustable with `recon_weight`) and the other related to the ability to fool the discriminator with samples generated by the VAE (adjustable with `gen_weight`).

The discriminator, on the other hand, is taught to classify both the reconstructions and the generated images as false.

    python AdvVAE.py --train --dataset octmnist

## Testing

This is related to the ability of the model to accurately reconstruct the input, which was encouraged during the training stage of the generator.

    python AdvVAE.py --test --dataset octmnist

## Sampling

This process is similar to the one described in [`VanillaVAE.md`](VanillaVAE.md).

    python AdvVAE.py --sample --dataset octmnist --checkpoint ./../../models/AdversarialVAE/AdvVAE.pt

## Outlier Detection

To detect out-of-distribution samples, we can either use the loss function as a way to produce an anomaly score, or the discriminator that was used for the adversarial training process.

    python AdvVAE.py --outlier_detection --dataset octmnist --out_dataset mnist --discriminator_checkpoint ./../../models/AdversarialVAE/Discriminator_octmnist.pt