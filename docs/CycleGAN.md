# Cycle Generative Adversarial Network (CycleGAN)

CycleGAN is a type of GAN designed for unsupervised image-to-image translation. Unlike traditional methods that require paired data for training, CycleGAN learns to translate images from one domain to another in the absence of paired examples. It accomplishes this by simultaneously training two generators and two discriminators in an adversarial manner.

## Parameters

| Parameter             | Description                                     | Default | Choices |
|-----------------------|-------------------------------------------------|---------|---------|
| `--train`             | train model                                     | `False` |         |
| `--test`              | test model                                      | `False` |         |
| `--batch_size`        | batch size                                      | `1`     |         |
| `--n_epochs`          | number of epochs                                | `200`   |         |
| `--lr`                | learning rate                                   | `0.0002`|         |
| `--decay`             | epoch to start linearly decaying the learning rate to 0 | `100` |         |
| `--sample_and_save_freq` | sample and save frequency                    | `5`     |         |
| `--dataset`           | dataset name                                    | `'horse2zebra'` | `'horse2zebra'`                                        |
| `--checkpoint_A`      | checkpoint A path                               | `None`  |         |
| `--checkpoint_B`      | checkpoint B path                               | `None`  |         |
| `--input_size`        | input size                                      | `128`   |         |
| `--in_channels`       | in channels                                     | `3`     |         |
| `--out_channels`      | out channels                                    | `3`     |         |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python CycGAN.py --help


## Training

CycleGAN introduces cycle consistency loss, which enforces the translated images to revert back to their original form when translated back to the original domain. This concept enables CycleGAN to learn mappings between domains in a self-supervised manner, making it particularly effective for tasks such as style transfer, object transfiguration, and domain adaptation without paired data. It couples this loss with an identity los, a reconstruction loss and the adversarial loss to train the generators.

    python CycGAN.py --train --dataset horse2zebra

## Test

Unfortunately, it is not possible to sample from a CycleGAN, we can only perform image-to-image translation. Therefore, if we input an image of domain A to the Generator that learnt how to map A->B, we can obtain a translated image.

    python CicGAN.py --test --dataset horse2zebra --checkpoint_A ./../../models/CycleGAN/CycGAN_horse2zebra_AB.pt --checkpoint_B ./../../models/CycleGAN/CycGAN_horse2zebra_BA.pt