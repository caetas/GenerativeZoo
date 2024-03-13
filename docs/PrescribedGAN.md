# Prescribed Generative Adversarial Networks

## Parameters

| Argument                  | Description                            | Default  | Choices                                                                 |
|---------------------------|----------------------------------------|----------|-------------------------------------------------------------------------|
| `--train`                 | Train model                            | `False`  |                                                                         |
| `--sample`                | Sample from model                      | `False`  |                                                                         |
| `--dataset`               | Dataset name                           | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn` |
| `--nz`                    | Size of the latent z vector            | `100`    |                                                                         |
| `--ngf`                   | Number of generator features           | `64`     |                                                                         |
| `--ndf`                   | Number of discriminator features       | `64`     |                                                                         |
| `--batch_size`            | Input batch size                       | `64`     |                                                                         |
| `--n_epochs`              | Number of epochs to train for          | `100`    |                                                                         |
| `--lrD`                   | Learning rate for discriminator        | `0.0002` |                                                                         |
| `--lrG`                   | Learning rate for generator            | `0.0002` |                                                                         |
| `--lrE`                   | Learning rate for encoder              | `0.0002` |                                                                         |
| `--beta1`                 | Beta1 for adam                         | `0.5`    |                                                                         |
| `--checkpoint`            | Checkpoint file for generator          | `None`   |                                                                         |
| `--discriminator_checkpoint` | Checkpoint file for discriminator    | `None`     |                                                                         |
| `--sigma_checkpoint`         | File for logsigma for the generator   | `None`     |                                                                         |
| `--num_gen_images`        | Number of images to generate           | `150`    |                                                                         |
| `--sigma_lr`              | Generator variance                     | `0.0002` |                                                                         |
| `--lambda_`               | Entropy coefficient                    | `0.01`   |                                                                         |
| `--sigma_min`             | Min value for sigma                    | `0.01`   |                                                                         |
| `--sigma_max`             | Max value for sigma                    | `0.3`    |                                                                         |
| `--logsigma_init`         | Initial value for log_sigma            | `-1.0`   |                                                                         |
| `--num_samples_posterior` | Number of samples from posterior       | `2`      |                                                                         |
| `--burn_in`               | HMC burn in                            | `2`      |                                                                         |
| `--leapfrog_steps`        | Number of leap frog steps for HMC      | `5`      |                                                                         |
| `--flag_adapt`            | Flag for HMC adaptation                | `1`      |                                                                         |
| `--delta`                 | Delta for HMC                          | `1.0`    |                                                                         |
| `--hmc_learning_rate`     | Learning rate for HMC                  | `0.02`   |                                                                         |
| `--hmc_opt_accept`        | HMC optimal acceptance rate            | `0.67`   |                                                                         |
| `--stepsize_num`          | Initial value for HMC stepsize         | `1.0`    |                                                                         |
| `--restrict_sigma`        | Whether to restrict sigma or not      | `0`      |                                                                         |
| `--sample_and_save_freq`  | Sample and save frequency              | `5`      |                                                                         |
| `--outlier_detection`     | Outlier detection                      | `False`  |                                                                         |
| `--out_dataset`           | Outlier dataset name                   | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn` |
