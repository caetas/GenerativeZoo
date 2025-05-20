# Vector-Quantized Generative Adversarial Network (VQ-GAN)

VQ-GAN combines vector quantization and adversarial training to produce high-fidelity image reconstructions and generation with a learned discrete latent space. The architecture consists of an encoder-decoder structure with a discrete codebook, trained jointly with a discriminator to enforce realism in the reconstructed images.

## Parameters

| Argument                 | Description                                             | Default     | Choices                                                                                                                                                                                                                |
| ------------------------ | ------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--train`                | Train model                                             | `False`     |                                                                                                                                                                                                                        |
| `--reconstruct`          | Reconstruct images using a trained model                | `False`     |                                                                                                                                                                                                                        |
| `--dataset`              | Dataset name                                            | `mnist`     | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `bloodmnist`, `dermamnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `retinamnist`, `svhn`, `tinyimagenet`, `imagenet`, `celeba` |
| `--batch_size`           | Batch size                                              | `256`       |                                                                                                                                                                                                                        |
| `--n_epochs`             | Number of training epochs                               | `100`       |                                                                                                                                                                                                                        |
| `--size`                 | Input image size (overrides dataset default)            | `None`      |                                                                                                                                                                                                                        |
| `--num_workers`          | Number of workers for Dataloader                        | `0`         |                                                                                                                                                                                                                        |
| `--warmup`               | Number of warm-up epochs before adversarial loss starts | `10`        |                                                                                                                                                                                                                        |
| `--channels`             | Base number of channels in encoder/decoder              | `64`        |                                                                                                                                                                                                                        |
| `--z_channels`           | Number of channels in latent space                      | `64`        |                                                                                                                                                                                                                        |
| `--ch_mult`              | Multipliers for channels at each resolution             | `[1, 2, 2]` |                                                                                                                                                                                                                        |
| `--num_res_blocks`       | Residual blocks per resolution                          | `2`         |                                                                                                                                                                                                                        |
| `--attn_resolutions`     | Resolutions at which to apply attention                 | `[16]`      |                                                                                                                                                                                                                        |
| `--dropout`              | Dropout rate                                            | `0.0`       |                                                                                                                                                                                                                        |
| `--double_z`             | Use double latent representation in encoder             | `False`     |                                                                                                                                                                                                                        |
| `--disc_start`           | Training step to start discriminator loss               | `10000`     |                                                                                                                                                                                                                        |
| `--disc_weight`          | Weight of discriminator loss in total loss              | `0.8`       |                                                                                                                                                                                                                        |
| `--codebook_weight`      | Weight of codebook loss                                 | `1.0`       |                                                                                                                                                                                                                        |
| `--n_embed`              | Number of embeddings in the codebook                    | `128`       |                                                                                                                                                                                                                        |
| `--embed_dim`            | Dimensionality of each embedding                        | `64`        |                                                                                                                                                                                                                        |
| `--remap`                | Path to remap indices for codebook                      | `None`      |                                                                                                                                                                                                                        |
| `--sane_index_shape`     | Use index shape compatible with quantizer               | `False`     |                                                                                                                                                                                                                        |
| `--checkpoint`           | Path to model checkpoint                                | `None`      |                                                                                                                                                                                                                        |
| `--colorize_nlabels`     | Number of semantic labels (for colorization tasks)      | `None`      |                                                                                                                                                                                                                        |
| `--lr`                   | Learning rate                                           | `4.5e-6`    |                                                                                                                                                                                                                        |
| `--no_wandb`             | Disable Weights & Biases logging                        | `False`     |                                                                                                                                                                                                                        |
| `--sample_and_save_freq` | Interval (in epochs) to sample and save reconstructions | `20`        |                                                                                                                                                                                                                        |

For further information on parameters, check [`util.py`](./../src/generativezoo/utils/util.py) or use:

```bash
python VQGAN.py --help
```

## Training

VQ-GAN is trained by minimizing a combination of reconstruction loss, codebook loss (via vector quantization), and adversarial loss. Adversarial training begins after a configurable number of warm-up epochs.

To train the model, run:

```bash
python VQGAN.py --train --dataset cifar10
```

## Reconstruction

To reconstruct images from the dataset using a trained model checkpoint:

```bash
python VQGAN.py --reconstruct --dataset cifar10 --checkpoint ./../../models/VQGAN/VQGAN_cifar10.pt
```
