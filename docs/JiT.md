# JiT

**This model supports `Accelerate` for Multi-GPU and Mixed Precision Training.**

## Parameters

| Argument                            | Default        | Help                                                          | Choices                                                                                                                                                                                                       |
|-------------------------------------|----------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--train`                           | `False`        | Train model                                                   |                                                                                                                                                                                                   |
| `--sample`                          | `False`        | Sample from model                                             |                                                                                                                                                                                                   |
| `--lr`                              | `1e-4`         | Learning rate                                                 |                                                                                                                                                                                                   |
| `--dataset`                         | `celeba`       | Dataset name                                                  | `mnist`, `cifar10`, `cifar100`, `places365`, `dtd`, `fashionmnist`, `chestmnist`, `bloodmnist`, `dermamnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `retinamnist`, `svhn`, `tinyimagenet`, `imagenet`, `celeba` |
| `--batch_size`                      | `256`          | Batch size                                                    |                                                                                                                                                                                                   |
| `--n_epochs`                        | `100`          | Number of epochs                                              |                                                                                                                                                                                                   |
| `--model`                           | `JiT-B/16`     | JiT model variant (e.g. JiT-B/16, JiT-B/32)                   |                                                                                                                                                                                                   |
| `--img_size`                        | `256`          | Input image size                                              |                                                                                                                                                                                                   |
| `--class_num`                       | `2`            | Number of classes (for label embedding)                       |                                                                                                                                                                                                   |
| `--attn_dropout`                    | `0.0`          | Attention dropout                                             |                                                                                                                                                                                                   |
| `--proj_dropout`                    | `0.0`          | Projection dropout                                            |                                                                                                                                                                                                   |
| `--label_drop_prob`                 | `0.1`          | Probability to drop labels (classifier-free guidance)         |                                                                                                                                                                                                   |
| `--P_mean`                          | `-0.8`         | Mean for timestep sampling (sigmoid space)                    |                                                                                                                                                                                                   |
| `--P_std`                           | `0.8`          | Std for timestep sampling (sigmoid space)                     |                                                                                                                                                                                                   |
| `--t_eps`                           | `1e-5`         | Epsilon for numerical stability in timesteps                  |                                                                                                                                                                                                   |
| `--noise_scale`                     | `1.0`          | Scale of initial noise for generation                         |                                                                                                                                                                                                   |
| `--ema_decay`                       | `0.9999`       | EMA decay (fast)                                              |                                                                                                                                                                                                   |
| `--sampling_method`                 | `euler`        | ODE sampling method                                           | `euler`, `heun`                                                                                                                                                                                 |
| `--num_sampling_steps`              | `50`           | Number of sampling steps for generation                       |                                                                                                                                                                                                   |
| `--cfg`                             | `2.9`          | Classifier-free guidance scale                                |                                                                                                                                                                                                   |
| `--interval_min`                    | `0.1`          | CFG interval min                                              |                                                                                                                                                                                                   |
| `--interval_max`                    | `1.0`          | CFG interval max                                              |                                                                                                                                                                                                   |
| `--num_workers`                     | `0`            | Number of workers for dataloader                              |                                                                                                                                                                                                   |
| `--weight_decay`                    | `0.0`          | Weight decay for Adam optimizer                               |                                                                                                                                                                                                   |
| `--snapshot`                        | `10`           | How many snapshots during training                            |                                                                                                                                                                                                   |
| `--no_wandb`                        | `False`        | Disable wandb logging                                         |                                                                                                                                                                                                   |
| `--sample_and_save_freq`            | `50`           | Sample and save frequency                                     |                                                                                                                                                                                                   |
| `--gradient_accumulation_steps`     | `1`            | Number of gradient accumulation steps                         |                                                                                                                                                                                                   |



You can find out more about the parameters by checking [`util.py`](./../src/generativezoo/utils/util.py) or by running the following command on the example script:

    python JiTransformer.py --help

## Training

You can train this model with the following command:

    accelerate launch JiTransformer.py --train --dataset cifar10

## Sampling

To sample, please provide the checkpoint:

    python JiTransformer.py --sample --dataset cifar10 --checkpoint ./../../models/JiT/FM_mnist.pt