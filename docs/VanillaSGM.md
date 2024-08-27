# Vanilla Score-Based Generative Model (Vanilla SGM)

Score-based Generative Models (SGMs) model the diffusion process using a Stochastic Differential Equation (SDE) instead of a Markov chain, as seen in traditional DDPMs. In this approach, the time-dependent score function is employed to construct the reverse-time SDE, which describes the process of removing noise from a sample to return it to the original data distribution. This reverse-time SDE is then solved numerically, often using techniques like Euler-Maruyama or Langevin dynamics, to generate samples from the original data distribution using samples from a simple prior distribution. To facilitate this process, a time-dependent score-based model is trained to approximate the score function. This model learns to estimate the score function at different timesteps, allowing for the construction of the reverse-time SDE and the generation of high-quality samples.

## Parameters

| Parameter             | Description                            | Default | Choices                                                      |
|-----------------------|----------------------------------------|---------|--------------------------------------------------------------|
| `--dataset`               | Dataset name                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                      | `False`  |                                                                                                                                                                              |
| `--out_dataset`       | outlier dataset name                   | `fashionmnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet` |
| `--batch_size`        | batch size                             | `128`   |                                                              |
| `--n_epochs`          | number of epochs                       | `100`   |                                                              |
| `--train`             | train model                            | `False` |                                                              |
| `--sample`            | sample model                           | `False` |                                                              |
| `--outlier_detection` | outlier detection                      | `False` |                                                              |
| `--model_channels`    | model channels                         | `[32, 64, 128, 256]` |                                                |
| `--atol`              | absolute tolerance                     | `1e-6`  |                                                              |
| `--rtol`              | relative tolerance                     | `1e-6`  |                                                              |
| `--eps`               | smallest timestep for numeric stability| `1e-3`  |                                                              |
| `--snr`               | signal to noise ratio                  | `0.16`  |                                                              |
| `--sample_timesteps`  | number of sampling timesteps           | `1000`  |                                                              |
| `--embed_dim`         | embedding dimension                    | `256`   |                                                              |
| `--lr`                | learning rate                          | `5e-4`  |                                                              |
| `--sigma`             | sigma                                  | `25.0`  |                                                              |
| `--checkpoint`        | checkpoint path                        | `None`  |                                                              |
| `--num_samples`       | number of samples                      | `16`    |                                                              |
| `--num_steps`         | number of steps                        | `500`   |                                                              |
| `--sampler_type`      | sampler type                           | `EM`  | `EM`, `PC`, `ODE`                                     |
| `--sample_and_save_freq` | sample and save frequency           | `10`    |                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python VanSGM.py --help

## Training

The Time-dependent score-based model is trained on random timesteps ranging from 0 to 1 to mimic a continuous time interval. The process is somewhat similar to what we have seen in DDPMs, although we don´t explicitly predict the noise that was added.

    python VanSGM.py --train --dataset svhn

## Sampling

To sample from SGMs, various numerical solvers can be employed. These solvers include classic methods like Euler-Maruyama and Predictor-Corrector, which are commonly used in stochastic differential equation (SDE) simulations. Additionally, ODE solvers can also be leveraged to make the process deterministic and reduce sampling times, although they sacrifice image quality. All these solvers are avilable via `--sampler_type`.

    python VanSGM.py --sample --dataset svhn --sampler_type PC --checkpoint ./../../models/VanillaSGM/VanSGM_svhn.pt

## Outlier Detection

To detect out-of-distribution samples, we use the loss function. In-distribution samples, resembling the training data, have low loss values, while out-of-distribution samples, unlike the training data, result in higher loss values.

    python VanSGM.py --outlier_detection --dataset svhn --out_dataset cifar10 --checkpoint ./../../models/VanillaSGM/VanSGM_svhn.pt