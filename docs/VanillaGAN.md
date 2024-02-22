# Vanilla Generative Adversarial Network (Vanilla GAN)

A Generative Adversarial Network (GAN) comprises two neural networks: a **Generator** and a **Discriminator**, engaged in a minimax game. The generator fabricates synthetic images out of a noisy input, while the discriminator evaluates the authenticity of these samples, distinguishing between real data and the generated ones. Through iterative training, the generator learns to produce increasingly realistic outputs that deceive the discriminator, while the discriminator enhances its ability to differentiate genuine from fake data.

## Parameters

| Parameter                   | Description                           | Default | Choices                                                         |
|-----------------------------|---------------------------------------|---------|-----------------------------------------------------------------|
| `--train`                 | train model                       | `False`|                                                                 |
| `--sample`                | sample from model                 | `False`|                                                                 |
| `--batch_size`            | batch size                        | `128` |                                                                 |
| `--dataset`               | dataset name                      | `'mnist'` | `'mnist'`, `'cifar10'`, `'fashionmnist'`, `'chestmnist'`, `'octmnist'`, `'tissuemnist'`, `'pneumoniamnist'`, `'svhn'`, `'cityscapes'` |
| `--n_epochs`              | number of epochs                  | `100` |                                                                 |
| `--lr`                    | learning rate                     | `0.0002` |                                                                 |
| `--beta1`                 | beta1                             | `0.5` |                                                                 |
| `--beta2`                 | beta2                             | `0.999` |                                                                |
| `--latent_dim`            | latent dimension                  | `100` |                                                                 |
| `--img_size`              | image size                        | `32`  |                                                                 |
| `--channels`              | channels                          | `1`   |                                                                 |
| `--sample_and_save_freq`       | sample interval                   | `5`   |                                                                 |
| `--checkpoint`            | checkpoint path                   | `None`|                                                                 |
| `--n_samples`             | number of samples                 | `9`   |                                                                 |
| `--d`                     | number of initial filters         | `128` |                                                                 |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VanGAN.py --help

## Training

Adversarial losses are used during training. The generator is encouraged to generate images that fool the discriminator into classifying them as real, while the discriminator is trained as a binary classifier to distinguish between real and generated images. The model can be trained using the following command:

    python VanGAN.py --train --dataset svhn

## Sampling

To sample from a GAN, you input a noisy latent vector of a predefined size into the generator network. This latent vector serves as a random seed that the generator uses to generate synthetic data samples.

    python VanGan.py --sample --dataset svhn --checkpoint ./../../models/VanillaGAN/VanGAN_svhn.pt