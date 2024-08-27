# Vector Quantised-Generative Adversarial Network + Transformer (VQ-GAN + Transformer)

The Vector Quantised-Generative Adversarial Network (VQ-GAN) is composed of an VQ-VAE that is trained adversarially to improve the quality of its reconstructions. To sample from this model, we train a Transformer model to generate plausible latent representations that can be decoded by the VQ-GAN.

## Parameters

| Argument                    | Description                                      | Default    | Choices                                                      |
|-----------------------------|--------------------------------------------------|------------|--------------------------------------------------------------|
| `--train`                   | Train model                                      | `False`    |                                                              |
| `--sample`                  | Sample model                                     | `False`    |                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--batch_size`              | Batch size                                       | `128`      |                                                              |
| `--n_epochs`                | Number of epochs for VQGAN                       | `100`      |                                                              |
| `--lr`                      | Learning rate VQGAN                              | `0.0002`   |                                                              |
| `--lr_d`                    | Learning rate discriminator                      | `0.0005`   |                                                              |
| `--adv_weight`              | Adversarial weight                               | `0.01`     |                                                              |
| `--perceptual_weight`       | Perceptual weight                                | `0.001`    |                                                              |
| `--lr_t`                    | Learning rate transformer                        | `0.0005`   |                                                              |
| `--n_epochs_t`              | Number of epochs transformer                     | `100`      |                                                              |
| `--num_res_layers`          | Number of residual layers                        | `2`        |                                                              |
| `--downsample_parameters`   | Downsample parameters                            | `2 4 1 1`  |                                                              |
| `--upsample_parameters`     | Upsample parameters                              | `2 4 1 1 0`|                                                              |
| `--num_channels`            | Number of channels                               | `256 256`  |                                                              |
| `--num_res_channels`        | Number of res channels                           | `256 256`  |                                                              |
| `--num_embeddings`          | Number of embeddings                             | `256`      |                                                              |
| `--embedding_dim`           | Embedding dimension                              | `32`       |                                                              |
| `--attn_layers_dim`         | Attn layers dim                                  | `96`       |                                                              |
| `--attn_layers_depth`       | Attn layers depth                                | `12`       |                                                              |
| `--attn_layers_heads`       | Attn layers heads                                | `8`        |                                                              |
| `--checkpoint`              | Checkpoint path to VQGAN                         | `None`     |                                                              |
| `--checkpoint_t`            | Checkpoint path to Transformer                   | `None`     |                                                              |
| `--num_samples`             | Number of samples                                | `16`       |                                                              |
| `--num_layers_d`            | Number of layers in discriminator                | `3`        |                                                              |
| `--num_channels_d`          | Number of channels in discriminator              | `64`       |                                                              |
| `--sample_and_save_freq`    | Sample and save frequency                        | `5`        |                                                              |
| `--outlier_detection`       | Outlier detection                                | `False`    |                                                              |
| `--out_dataset`             | Outlier dataset name                             | `fashionmnist`| `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VQGAN_T.py --help

## Training

The VQGAN and the Transformer can both be trained with a single command:

    python VQGAN_T.py --train --dataset svhn

## Sampling

For sampling you must provide both checkpoints:

    python VQGAN_T.py --sample --dataset svhn --checkpoint ./../../models/VQGAN_Transformer/VQGAN_svhn.pt --checkpoint_t ./../../models/VQGAN_Transformer/Transformer_svhn.pt
