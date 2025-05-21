# MaskGIT (Masked Generative Image Transformer)

**MaskGIT** is a bidirectional image generation model based on iterative masked token modeling. It operates on discrete tokens produced by a **VQ-GAN**, where generation is performed through a parallel decoding process using a transformer.

**Note:** A **trained VQ-GAN** is required to encode images into discrete tokens. MaskGIT is trained on these tokens using a masked token prediction objective.

## Parameters

| Argument                 | Description                                    | Default       | Choices                                                                                                                                                                                                                |
| ------------------------ | ---------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--train`                | Train MaskGIT                                  | `False`       |                                                                                                                                                                                                                        |
| `--sample`               | Sample from trained model                      | `False`       |                                                                                                                                                                                                                        |
| `--dataset`              | Dataset name                                   | `mnist`       | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `bloodmnist`, `dermamnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `retinamnist`, `svhn`, `tinyimagenet`, `imagenet`, `celeba` |
| `--batch_size`           | Batch size                                     | `256`         |                                                                                                                                                                                                                        |
| `--n_epochs`             | Number of training epochs                      | `100`         |                                                                                                                                                                                                                        |
| `--size`                 | Override dataset image size                    | `None`        |                                                                                                                                                                                                                        |
| `--num_workers`          | DataLoader worker threads                      | `0`           |                                                                                                                                                                                                                        |
| `--warmup`               | Warmup epochs                                  | `10`          |                                                                                                                                                                                                                        |
| `--channels`             | Base encoder/decoder channels (VQ-GAN)         | `64`          |                                                                                                                                                                                                                        |
| `--z_channels`           | Latent space channels                          | `64`          |                                                                                                                                                                                                                        |
| `--ch_mult`              | Channel multipliers per resolution             | `[1,2,2]`     |                                                                                                                                                                                                                        |
| `--num_res_blocks`       | Residual blocks per resolution                 | `2`           |                                                                                                                                                                                                                        |
| `--attn_resolutions`     | Attention resolutions                          | `[16]`        |                                                                                                                                                                                                                        |
| `--dropout`              | Dropout in encoder/decoder (VQ-GAN)            | `0.0`         |                                                                                                                                                                                                                        |
| `--double_z`             | Use double latent encoding                     | `False`       |                                                                                                                                                                                                                        |
| `--disc_start`           | Iteration to start discriminator loss (VQ-GAN) | `10000`       |                                                                                                                                                                                                                        |
| `--disc_weight`          | Discriminator loss weight                      | `0.8`         |                                                                                                                                                                                                                        |
| `--codebook_weight`      | Codebook loss weight (VQ-GAN)                  | `1.0`         |                                                                                                                                                                                                                        |
| `--n_embed`              | Codebook size                                  | `128`         |                                                                                                                                                                                                                        |
| `--embed_dim`            | Embedding dim for VQ-GAN                       | `64`          |                                                                                                                                                                                                                        |
| `--embed_dim_t`          | Embedding dim for transformer                  | `64`          |                                                                                                                                                                                                                        |
| `--remap`                | Codebook index remapping                       | `None`        |                                                                                                                                                                                                                        |
| `--sane_index_shape`     | Use stable index shape in quantizer            | `False`       |                                                                                                                                                                                                                        |
| `--checkpoint_vae`       | Path to pretrained VQ-GAN checkpoint           | `None`        |                                                                                                                                                                                                                        |
| `--colorize_nlabels`     | Labels used for colorization                   | `None`        |                                                                                                                                                                                                                        |
| `--lr`                   | Learning rate                                  | `1e-4`        |                                                                                                                                                                                                                        |
| `--no_wandb`             | Disable WandB logging                          | `False`       |                                                                                                                                                                                                                        |
| `--sample_and_save_freq` | Sampling frequency during training             | `20`          |                                                                                                                                                                                                                        |
| `--betas`                | Adam optimizer betas                           | `[0.9, 0.95]` |                                                                                                                                                                                                                        |
| `--weight_decay`         | Weight decay                                   | `0.1`         |                                                                                                                                                                                                                        |

---

### MaskGIT-Specific Parameters

| Argument        | Description                                                   | Default  |
| --------------- | ------------------------------------------------------------- | -------- |
| `--cfg_w`       | Classifier-free guidance weight                               | `3.0`    |
| `--r_temp`      | Gumbel noise temperature for resampling                       | `4.5`    |
| `--sm_temp`     | Temperature before softmax in prediction                      | `1.0`    |
| `--drop-label`  | Drop rate for CFG conditioning                                | `0.1`    |
| `--hidden_dim`  | Transformer hidden dimension                                  | `128`    |
| `--heads`       | Number of attention heads                                     | `8`      |
| `--depth`       | Transformer depth (number of layers)                          | `10`     |
| `--mlp_dim`     | MLP dimension                                                 | `384`    |
| `--dropout_t`   | Dropout in transformer                                        | `0.1`    |
| `--step`        | Number of iterative steps during sampling                     | `8`      |
| `--sched_mode`  | Mask scheduling strategy (e.g., `cosine`, `linear`, `arccos`) | `arccos` |
| `--mask-value`  | Integer used to mask tokens during inference                  | `None`   |
| `--n_classes`   | Number of classes (for conditional generation)                | `10`     |
| `--num_samples` | Number of generated samples                                   | `16`     |

## Usage

### Train MaskGIT

```bash
python MaskGIT.py --train --dataset cifar10 --checkpoint_vae ./../../models/VQGAN/VQGAN_cifar10.pt
```

### Sample from MaskGIT

```bash
python MaskGIT.py --sample --dataset cifar10 --checkpoint_vae ./../../models/VQGAN/VQGAN_cifar10.pt
```

### Help

```bash
python MaskGIT.py --help
```