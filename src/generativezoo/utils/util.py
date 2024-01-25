from pathlib import Path
import numpy as np
import argparse


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def set_seed(seed: int = 42) -> None:
    """Set random seed for numpy.

    https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    """
    rng = np.random.default_rng(seed)
    return rng

def parse_args_VanillaSGM():
    argparser = argparse.ArgumentParser()
    # show choices: mnist | cifar10 | fashionmnist | chestmnist | octmnist | tissuemnist | pneumoniamnist | svhn
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    argparser.add_argument('--sigma', type=float, default=25.0, help='sigma')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--num_steps', type=int, default=500, help='number of steps')
    argparser.add_argument('--sampler_type', type=str, default='Euler-Maruyama', help='sampler type', choices=['Euler-Maruyama', 'PC', 'ODE'])
    return argparser.parse_args()
# EOF
