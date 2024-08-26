# NCSNv2

This work provides a new theoretical analysis of learning and sampling from score-based models in high dimensional spaces, explaining existing failure modes and motivating new solutions that generalize across datasets. To enhance stability, it also proposes to maintain an exponential moving average of model weights.

## Parameters

| Argument                    | Description                                      | Default     | Choices                                                      |
|-----------------------------|--------------------------------------------------|-------------|--------------------------------------------------------------|
| `--train`                   | Train model                                      | `False`     |                                                              |
| `--sample`                  | Sample from model                                | `False`     |                                                              |
| `--dataset`               | Dataset name                                       | `mnist`  | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `tinyimagenet`, `cifar100`, `places365`, `dtd`, `imagenet`            |
| `--no_wandb`              | Disable Wandb                                      | `False`  |                                                                                                                                                                              |
| `--batch_size`              | Batch size                                       | `128`       |                                                              |
| `--n_epochs`                | Number of epochs                                 | `100`       |                                                              |
| `--lr`                      | Learning rate                                    | `0.0002`    |                                                              |
| `--nf`                      | Number of filters                                | `128`       |                                                              |
| `--act`                     | Activation                                       | `elu`     | `relu`, `elu`, `swish`                                 |
| `--centered`                | Centered                                         | `False`     |                                                              |
| `--sigma_min`               | Min value for sigma                              | `0.01`      |                                                              |
| `--sigma_max`               | Max value for sigma                              | `50`        |                                                              |
| `--num_scales`              | Number of scales                                 | `232`       |                                                              |
| `--normalization`           | Normalization                                    | `InstanceNorm++` | `InstanceNorm`, `GroupNorm`, `VarianceNorm`, `InstanceNorm++` |
| `--num_classes`             | Number of classes                                | `10`        |                                                              |
| `--ema_decay`               | EMA decay                                        | `0.999`     |                                                              |
| `--continuous`              | Continuous                                       | `False`     |                                                              |
| `--reduce_mean`             | Reduce mean                                      | `False`     |                                                              |
| `--likelihood_weighting`    | Likelihood weighting                             | `False`     |                                                              |
| `--beta1`                   | Beta1                                            | `0.9`       |                                                              |
| `--beta2`                   | Beta2                                            | `0.999`     |                                                              |
| `--weight_decay`            | Weight decay                                     | `0.0`       |                                                              |
| `--warmup`                  | Warmup                                           | `0`         |                                                              |
| `--grad_clip`               | Grad clip                                        | `-1.0`      |                                                              |
| `--sample_and_save_freq`    | Sample and save frequency                        | `5`         |                                                              |
| `--sampler`                 | Sampler name                                     | `pc`      | `pc`, `ode`                                              |
| `--predictor`               | Predictor                                        | `none`    | `none`, `em`, `rd`, `as`                             |
| `--corrector`               | Corrector                                        | `ald`     | `none`, `l`, `ald`                                     |
| `--snr`                     | Signal to noise ratio                            | `0.176`     |                                                              |
| `--n_steps`                 | Number of steps                                  | `5`         |                                                              |
| `--probability_flow`        | Probability flow                                 | `False`     |                                                              |
| `--noise_removal`           | Noise removal                                    | `False`     |                                                              |
| `--checkpoint`              | Checkpoint path to VQVAE                         | `None`     |                                                              |
| `--num_workers`   | Number of workers for Dataloader   | `0`     |                                                              |

You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python NCSNv2.py --help

## Training

The model can be trained with:

    python .\NCSNv2.py --train --nf 32 --noise_removal

## Sampling

For sampling you must provide the checkpoint:

    python .\NCSNv2.py --sample --nf 32 --noise_removal --checkpoint ./../../models/NCSNv2/NCSNv2_mnist.pt