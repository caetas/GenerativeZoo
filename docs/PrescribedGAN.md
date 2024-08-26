# Prescribed Generative Adversarial Networks (PresGANs)

PresGANs add noise to the output of a density network and optimize an entropy-regularized adversarial loss. The added noise renders tractable approximations of the predictive log-likelihood and stabilizes the training procedure. The entropy regularizer encourages PresGANs to capture all the modes of the data distribution.

## Parameters

| Argument                  | Description                                        | Default  | Choices                                                                                                  |
|---------------------------|----------------------------------------------------|----------|----------------------------------------------------------------------------------------------------------|
| `--train`                 | Train model                                        | `False`  |                                                                                                          |
| `--sample`                | Sample from model                                  | `False`  |                                                                                                          |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--nz`                    | Size of the latent z vector                        | `100`    |                                                                                                          |
| `--ngf`                   |                                                    | `64`     |                                                                                                          |
| `--ndf`                   |                                                    | `64`     |                                                                                                          |
| `--batch_size`            | Input batch size                                   | `64`     |                                                                                                          |
| `--n_epochs`              | Number of epochs to train for                      | `100`    |                                                                                                          |
| `--lrD`                   | Learning rate for discriminator                   | `0.0002` |                                                                                                          |
| `--lrG`                   | Learning rate for generator                       | `0.0002` |                                                                                                          |
| `--lrE`                   | Learning rate                                      | `0.0002` |                                                                                                          |
| `--beta1`                 | Beta1 for adam                                     | `0.5`    |                                                                                                          |
| `--checkpoint`            | Checkpoint file for generator                      | `None`   |                                                                                                          |
| `--discriminator_checkpoint` | Checkpoint file for discriminator                | `None`   |                                                                                                          |
| `--sigma_checkpoint`      | File for logsigma for the generator               | `None`   |                                                                                                          |
| `--num_gen_images`        | Number of images to generate for inspection        | `16`     |                                                                                                          |
| `--sigma_lr`              | Generator variance                                 | `0.0002` |                                                                                                          |
| `--lambda_`               | Entropy coefficient                               | `0.01`   |                                                                                                          |
| `--sigma_min`             | Min value for sigma                               | `0.01`   |                                                                                                          |
| `--sigma_max`             | Max value for sigma                               | `0.3`    |                                                                                                          |
| `--logsigma_init`         | Initial value for log_sigma_sian                  | `-1.0`   |                                                                                                          |
| `--num_samples_posterior` | Number of samples from posterior                  | `2`      |                                                                                                          |
| `--burn_in`               | Hmc burn in                                        | `2`      |                                                                                                          |
| `--leapfrog_steps`        | Number of leap frog steps for hmc                 | `5`      |                                                                                                          |
| `--flag_adapt`            | `0` or `1`                                         | `1`      |                                                                                                          |
| `--delta`                 | Delta for hmc                                      | `1.0`    |                                                                                                          |
| `--hmc_learning_rate`     | Lr for hmc                                         | `0.02`   |                                                                                                          |
| `--hmc_opt_accept`        | Hmc optimal acceptance rate                       | `0.67`   |                                                                                                          |
| `--stepsize_num`          | Initial value for hmc stepsize                    | `1.0`    |                                                                                                          |
| `--restrict_sigma`        | Whether to restrict sigma or not                  | `0`      |                                                                                                          |
| `--sample_and_save_freq`  | Sample and save frequency                         | `5`      |                                                                                                          |
| `--outlier_detection`     | Outlier detection                                  | `False`  |                                                                                                          |
| `--out_dataset`           | Outlier dataset name                              | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python PresGAN.py --help

## Training

The PresGAN can be trained in a similar fashion to other GANs in the zoo:

    python PresGAN.py --train --dataset tinyimagenet --restrict_sigma 1 --sigma_min 1e-3 --sigma_max 0.3 --lambda 5e-4 --nz 1024

## Sampling

For sampling you must provide the generator checkpoint:

    python PresGAN.py --sample --dataset tinyimagenet --nz 1024 --checkpoint ./../../models/PrescribedGAN/PresGAN_tinyimagenet.pt

## Outlier Detection

To perform outlier detection you must provide the discriminator checkpoint and the sigma checkpoint:

    python PresGAN.py --sample --dataset tinyimagenet --out_dataset cifar10 --nz 1024 --discriminator_checkpoint ./../../models/PrescribedGAN/PresDisc_tinyimagenet.pt --sigma_checkpoint ./../../models/PrescribedGAN/PresSigma_tinyimagenet.pt

