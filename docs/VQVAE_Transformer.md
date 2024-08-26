# Vector Quantised-Variational AutoEncoder + Transformer (VQ-VAE + Transformer)

The Vector Quantised-Variational AutoEncoder (VQ-VAE), differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static. To sample from this model, we train a Transformer model to generate plausible latent representations that can be decoded by the VQ-VAE.

## Parameters

| Argument                    | Description                                      | Default    | Choices                                                      |
|-----------------------------|--------------------------------------------------|------------|--------------------------------------------------------------|
| `--train`                   | Train model                                      | `False`    |                                                              |
| `--sample`                  | Sample model                                     | `False`    |                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--batch_size`              | Batch size                                       | `128`      |                                                              |
| `--n_epochs`                | Number of epochs for VQVAE                       | `100`      |                                                              |
| `--lr`                      | Learning rate VQVAE                              | `0.0002`   |                                                              |
| `--lr_t`                    | Learning rate transformer                        | `0.0002`   |                                                              |
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
| `--checkpoint`              | Checkpoint path to VQVAE                         | `None`     |                                                              |
| `--checkpoint_t`            | Checkpoint path to Transformer                   | `None`     |                                                              |
| `--num_samples`             | Number of samples                                | `16`       |                                                              |
| `--sample_and_save_freq`    | Sample and save frequency                        | `5`        |                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VQVAE_T.py --help

## Training

The VQVAE and the Transformer can both be trained with a single command:

    python VQVAE_T.py --train --dataset svhn

## Sampling

For sampling you must provide both checkpoints:

    python VQVAE_T.py --sample --dataset svhn --checkpoint ./../../models/VQVAE_Transformer/VQVAE_svhn.pt --checkpoint_t ./../../models/VQVAE_Transformer/Transformer_svhn.pt