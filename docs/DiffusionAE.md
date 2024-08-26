# Diffusion Autoencoder (DiffAE)

The Diffusion Autoencoder (DiffAE) is a model that learns to encode images into a latent space using an encoder and then utilizes this latent representation to guide the image generation process through a diffusion network. By jointly training the encoder and diffusion network, the diffusion autoencoder achieves effective latent space representation learning and image generation, facilitating tasks such as image reconstruction and image manipulation.

## Parameters

| Parameter             | Description                            | Default | Choices                                                      |
|-----------------------|----------------------------------------|---------|--------------------------------------------------------------|
| `--train`             | train model                            | `False` |                                                              |
| `--manipulate`        | manipulate latents                     | `False` |                                                              |
| `--batch_size`        | batch size                             | `16`    |                                                              |
| `--n_epochs`          | number of epochs                       | `100`   |                                                              |
| `--lr`                | learning rate                          | `0.001` |                                                              |
| `--timesteps`         | number of timesteps                    | `1000`  |                                                              |
| `--sample_timesteps`  | number of timesteps for sampling       | `100`   |                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--checkpoint`        | checkpoint path                        | `None`  |                                                              |
| `--embedding_dim`     | embedding dimension                    | `512`   |                                                              |
| `--model_channels`    | model channels                         | `[64, 128, 256]` |                                                |
| `--attention_levels`  | attention levels                       | `[False, True, True]` |                                           |
| `--num_res_blocks`    | number of res blocks                   | `1`     |                                                              |
| `--sample_and_save_freq` | sample and save frequency           | `10`    |                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python DAE.py --help

## Training

During training, the diffusion autoencoder leverages the noise prediction capability of the diffusion network to simultaneously train both the encoder and diffusion network. This training process presents opportunities for future optimizations by modifying the training objective to include latent space representation losses.

    python DAE.py --train --dataset pneumoniamnist

## Manipulate Images

While direct sampling from the model may not be feasible, an alternative approach involves training binary classifiers on the embeddings produced by the encoders, using class or feature labels. These classifiers can then be leveraged to manipulate the latent space, enabling the control of specific features within generated images. This technique allows for targeted manipulation of image features.

    python DAE.py --manipulate --dataset pneumoniamnist --checkpoint ./../../models/DiffusionAE/DiffAE_pneumoniamnist.pt