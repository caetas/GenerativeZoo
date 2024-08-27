# Conditional Generative Adversarial Network (cGAN)

A Conditional Generative Adversarial Network (cGAN) is an extension of the traditional GAN framework where additional conditioning information, typically in the form of class labels or embeddings, is provided to both the generator and the discriminator. This allows for the generation of samples conditioned on specific attributes or classes, enhancing control over the generated outputs.

## Parameters

| Parameter         | Description                        | Default | Choices                                                      |
|-------------------|------------------------------------|---------|--------------------------------------------------------------|
| `--train`         | train model                        | `False` |                                                              |
| `--sample`        | sample from model                  | `False` |                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--batch_size`    | batch size                         | `128`   |                                                              |
| `--n_epochs`      | number of epochs                   | `100`   |                                                              |
| `--lr`            | learning rate                      | `0.0002`|                                                              |
| `--beta1`         | beta1                              | `0.5`   |                                                              |
| `--beta2`         | beta2                              | `0.999` |                                                              |
| `--latent_dim`    | latent dimension                   | `100`   |                                                              |
| `--n_classes`     | number of classes                  | `10`    |                                                              |
| `--img_size`      | image size                         | `32`    |                                                              |
| `--channels`      | channels                           | `1`     |                                                              |
| `--sample_and_save_freq` | sample interval                  | `5`     |                                                              |
| `--checkpoint`    | checkpoint path                    | `None`  |                                                              |
| `--n_samples`     | number of samples                  | `9`     |                                                              |
| `--d`             | number of initial filters          | `128`   |                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python CondGAN.py --help

## Training

The training process is similar to the one mentioned in [`VanillaGAN.md`](VanillaGAN.md), but with the inclusion of the aforementioned embeddings.

    python CondGAN.py --train --dataset mnist --n_classes 10

## Sampling

The sampling process is also similar but it requires to also include the class-related embedding and not only a noisy latent sample:

    python CondGAN.py --train --dataset mnist --n_classes 10 --checkpoint ./../../models/ConditionalGAN/CondGAN_mnist.pt