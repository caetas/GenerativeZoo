# Glow

## Parameters

| Argument             | Description                           | Default | Choices                                              |
|----------------------|---------------------------------------|---------|------------------------------------------------------|
| `--train`            | Train model                           | `False` |                                                      |
| `--sample`           | Sample from model                     | `False` |                                                      |
| `--outlier_detection`| Outlier detection                     | `False` |                                                      |
| `--dataset`          | Dataset name                          | `mnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `cityscapes` |
| `--out_dataset`      | Outlier dataset name                  | `mnist` | `mnist`, `cifar10`, `fashionmnist`, `chestmnist`, `octmnist`, `tissuemnist`, `pneumoniamnist`, `svhn`, `cityscapes` |
| `--batch_size`       | Batch size                            | `128`   |                                                      |
| `--n_epochs`         | Number of epochs                      | `100`   |                                                      |
| `--lr`               | Learning rate                         | `0.0002`|                                                      |
| `--hidden_channels`  | Hidden channels                       | `64`    |                                                      |
| `--K`                | Number of layers per block            | `8`     |                                                      |
| `--L`                | Number of blocks                      | `3`     |                                                      |
| `--actnorm_scale`    | Act norm scale                        | `1.0`   |                                                      |
| `--flow_permutation` | Flow permutation                      |`invconv`| `invconv`, `shuffle`, `reverse`                      |
| `--flow_coupling`    | Flow coupling                         |`affine` | `additive`, `affine`                                 |
| `--LU_decomposed`    | Train with LU decomposed 1x1 convs    |`False`  |                                                      |
| `--learn_top`        | Learn top layer (prior)               | `False` |                                                      |
| `--y_condition`      | Class Conditioned Glow                | `False` |                                                      |
| `--y_weight`         | Weight of class condition             | `0.01`  |                                                      |
| `--num_classes`      | Number of classes                     | `10`    |                                                      |
| `--sample_and_save_freq` | Sample and save frequency         | `5`     |                                                      |
| `--checkpoint`       | Checkpoint path                       | `None`  |                                                      |
| `--n_bits`           | Number of bits                        | `8`     |                                                      |
