# Vanilla Generative Adversarial Network (Vanilla GAN)

A Generative Adversarial Network (GAN) comprises two neural networks: a **Generator** and a **Discriminator**, engaged in a minimax game. The generator fabricates synthetic images out of a noisy input, while the discriminator evaluates the authenticity of these samples, distinguishing between real data and the generated ones. Through iterative training, the generator learns to produce increasingly realistic outputs that deceive the discriminator, while the discriminator enhances its ability to differentiate genuine from fake data.

## Parameters

| Argument                  | Description                                        | Default  | Choices                                                                                                                                                                      |
|---------------------------|----------------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--train`                 | Train model                                        | `False`  |                                                                                                                                                                              |
| `--sample`                | Sample from model                                  | `False`  |                                                                                                                                                                              |
| `--outlier_detection`     | Outlier detection                                  | `False`  |                                                                                                                                                                              |
| `--batch_size`            | Batch size                                         | `128`    |                                                                                                                                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--out_dataset`           | Outlier dataset name                               | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`      |
| `--n_epochs`              | Number of epochs                                   | `100`    |                                                                                                                                                                              |
| `--lrg`                   | Learning rate generator                            | `0.0002` |                                                                                                                                                                              |
| `--lrd`                   | Learning rate discriminator                        | `0.0002` |                                                                                                                                                                              |
| `--beta1`                 | Beta1                                              | `0.5`    |                                                                                                                                                                              |
| `--beta2`                 | Beta2                                              | `0.999`  |                                                                                                                                                                              |
| `--latent_dim`            | Latent dimension                                   | `100`    |                                                                                                                                                                              |
| `--img_size`              | Image size                                         | `32`     |                                                                                                                                                                              |
| `--channels`              | Channels                                           | `1`      |                                                                                                                                                                              |
| `--sample_and_save_freq`  | Sample interval                                    | `5`      |                                                                                                                                                                              |
| `--checkpoint`            | Checkpoint path                                    | `None`   |                                                                                                                                                                              |
| `--discriminator_checkpoint` | Discriminator checkpoint path                   | `None`   |                                                                                                                                                                              |
| `--n_samples`             | Number of samples                                  | `9`      |                                                                                                                                                                              |
| `--d`                     | d                                                  | `128`    |                                                                                                                                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VanGAN.py --help

## Training

Adversarial losses are used during training. The generator is encouraged to generate images that fool the discriminator into classifying them as real, while the discriminator is trained as a binary classifier to distinguish between real and generated images. The model can be trained using the following command:

    python VanGAN.py --train --dataset svhn

## Sampling

To sample from a GAN, you input a noisy latent vector of a predefined size into the generator network. This latent vector serves as a random seed that the generator uses to generate synthetic data samples.

    python VanGan.py --sample --dataset svhn --checkpoint ./../../models/VanillaGAN/VanGAN_svhn.pt

## Outlier Detection

To perform outlier detection, only the Discriminator will be used:

    python VanGan.py --outlier_detection --dataset svhn --out_dataset cifar10 --discriminator_checkpoint ./../../models/VanillaGAN/VanDisc_svhn.pt