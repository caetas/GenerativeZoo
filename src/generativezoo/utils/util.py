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

def parse_args_DDPM():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--timesteps', type=int, default=300, help='number of timesteps')
    argparser.add_argument('--beta_start', type=float, default=0.0001, help='beta start')
    argparser.add_argument('--beta_end', type=float, default=0.02, help='beta end')
    argparser.add_argument('--sample_and_save_freq', type=int, default=10, help='sample and save frequency')
    argparser.add_argument('--device', type=str, default='cuda', help='device')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--ddpm', type=float, default=1.0, help='ddim sampling is 0.0, pure ddpm is 1.0')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--num_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--loss_type', type=str, default='huber', help='loss type', choices=['huber','l2', 'l1'])
    argparser.add_argument('--sample_timesteps', type=int, default=300, help='number of timesteps')
    return argparser.parse_args()

def parse_args_CDDPM():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action='store_true', default=False, help='train model')
    argparser.add_argument('--sample', action='store_true', default=False, help='sample model')
    argparser.add_argument('--outlier_detection', action='store_true', default=False, help='outlier detection')
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--timesteps', type=int, default=500, help='number of timesteps')
    argparser.add_argument('--beta_start', type=float, default=0.0001, help='beta start')
    argparser.add_argument('--beta_end', type=float, default=0.02, help='beta end')
    argparser.add_argument('--device', type=str, default='cuda', help='device')
    argparser.add_argument('--dataset', type=str, default='mnist', help='dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--ddpm', type=float, default=1.0, help='ddim sampling is 0.0, pure ddpm is 1.0')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--out_dataset', type=str, default='fashionmnist', help='outlier dataset name', choices=['textile','toothbrush','bottle','mnist', 'cifar10', 'fashionmnist', 'chestmnist', 'octmnist', 'tissuemnist', 'pneumoniamnist', 'svhn'])
    argparser.add_argument('--sample_timesteps', type=int, default=500, help='number of timesteps')
    argparser.add_argument('--n_features', type=int, default=128, help='number of features')
    argparser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    argparser.add_argument('--sample_and_save_freq', type=int, default=10, help='sample and save frequency')
    argparser.add_argument('--drop_prob', type=float, default=0.1, help='dropout probability')
    argparser.add_argument('--guide_w', type=float, default=0.5, help='guide weight')
    return argparser.parse_args()

# EOF
