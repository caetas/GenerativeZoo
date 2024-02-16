# GenerativeZoo

[![Python](https://img.shields.io/badge/python-3.9+-informational.svg)](https://www.python.org/downloads/release/python-3918/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=black)](https://pycqa.github.io/isort)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://mkdocstrings.github.io)
[![wandb](https://img.shields.io/badge/tracking-wandb-blue)](https://wandb.ai/site)
[![dvc](https://img.shields.io/badge/data-dvc-9cf)](https://dvc.org)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

A collection of generative algorithms and techniques implemented in Python.

## Prerequisites

You will need:

- `python` (see `pyproject.toml` for full version)
- `Git`
- `Make`
- a `.secrets` file with the required secrets and credentials
- load environment variables from `.env`

## Installation

Clone this repository (requires git ssh keys)

    git clone --recursive git@github.com:caetas/GenerativeZoo.git
    cd generativezoo

Install dependencies

    conda create -y -n python3.9 python=3.9
    conda activate python3.9

or if environment already exists

    conda env create -f environment.yml
    conda activate python3.9

### On Linux

And then setup all virtualenv using make file recipe

    (python3.9) $ make setup-all

You might be required to run the following command once to setup the automatic activation of the conda environment and the virtualenv:

    direnv allow

Feel free to edit the [`.envrc`](.envrc) file if you prefer to activate the environments manually.

### On Windows

You can setup the virtualenv by running the following commands:

    python -m venv .venv-dev
    source .venv-dev/bin/activate
    python -m pip install --upgrade pip setuptools
    python -m pip install -r requirements/requirements.txt

To run the code please remember to always activate both environments:

    conda activate python3.9
    source .venv-dev/bin/activate

## Models

### Implemented Models

The listed models are already implemented and fully integrated in the model zoo.

#### VAEs

- Vanilla VAE [`Paper`](https://arxiv.org/abs/1312.6114)|[`Code`](src/generativezoo/models/VAE/VanillaVAE.py)|[`Script`](src/generativezoo/VanVAE.py)|[`Documentation`](models/VanillaVAE.md)
- Conditional VAE [`Paper`](https://openreview.net/forum?id=rJWXGDWd-H)|[`Code`](src/generativezoo/models/VAE/ConditionalVAE.py)|[`Script`](src/generativezoo//CondVAE.py)|[`Documentation`](models/ConditionalVAE.md)

#### GANs

- Adversarial VAE [`Paper`](https://arxiv.org/abs/1511.05644)|[`Code`](src/generativezoo/models/GANs/AdversarialVAE.py)|[`Script`](src/generativezoo/AdvVAE.py)|[`Documentation`](models/AdversarialVAE.md)
- Vanilla GAN [`Paper`](https://arxiv.org/abs/1511.06434)|[`Code`](src/generativezoo/models/GANs/VanillaGAN.py)|[`Script`](src/generativezoo/VanGAN.py)|[`Documentation`](models/VanillaGAN.md)
- Conditional GAN [`Paper`](https://arxiv.org/abs/1411.1784)|[`Code`](src/generativezoo/models/GANs/ConditionalGAN.py)|[`Script`](src/generativezoo/CondGAN.py)|[`Documentation`](models/ConditionalGAN.md)
- CycleGAN [`Paper`](https://arxiv.org/abs/1703.10593)|[`Code`](src/generativezoo/models/GANs/CycleGAN.py)|[`Script`](src/generativezoo/CycGAN.py)|[`Documentation`](models/CycleGAN.md)

#### DDPMs

- Vanilla DDPM [`Paper`](https://arxiv.org/abs/2006.11239)|[`Code`](src/generativezoo/models/DDPM/VanillaDDPM.py)|[`Script`](src/generativezoo/DDPM.py)|[`Documentation`](models/VanillaDDPM.md)
- Conditional DDPM [`Paper`](https://arxiv.org/abs/2207.12598)|[`Code`](src/generativezoo/models/DDPM/ConditionalDDPM.py)|[`Script`](src/generativezoo/CondDDPM.py)
- Diffusion AE [`Paper`](https://arxiv.org/abs/2111.15640)|[`Code`](src/generativezoo/models/DDPM/MONAI_DiffAE.py)|[`Script`](src/generativezoo/DAE.py)

#### SGMs

- Vanilla SGM [`Paper`](https://arxiv.org/abs/2006.09011)|[`Code`](src/generativezoo/models/SGM/VanillaSGM.py)|[`Script`](src/generativezoo/SGM.py)

### Future Models

These models are currently under development and will be added to the repository in the future.

#### VAEs

- [ ] Hierarchical VAE [`Paper`](https://arxiv.org/abs/2007.03898)
- [ ] VQ-VAE [`Paper`](https://arxiv.org/abs/1711.00937)

#### GANs

- [ ] VQ-GAN [`Paper`](https://arxiv.org/abs/2012.09841)

#### SGMs

- [ ] NCSN [`Paper`](https://arxiv.org/abs/1907.05600)
- [ ] NCSN++ [`Paper`](https://openreview.net/forum?id=PxTIG12RRHS)

#### Autoregressive

- [ ] VQ-VAE + Transformer [`Paper`](https://arxiv.org/abs/2012.09841)
- [ ] VQ-VAE + Mamba

#### Flow-Based Models

- [ ] RealNVP [`Paper`](https://arxiv.org/abs/1605.08803)
- [ ] Flow++ [`Paper`](https://arxiv.org/abs/1902.00275)

## Datasets

The following datasets are ready to be used to train and sample from the provided models. They are automatically downloaded when you try to use them for the first time.

### Grayscale

- MNIST [`Source`](https://ieeexplore.ieee.org/document/726791)
- FashionMNIST [`Source`](https://arxiv.org/abs/1708.07747)
- ChestMNIST++ [`Source`](https://www.nature.com/articles/s41597-022-01721-8)
- OctMNIST++ [`Source`](https://www.nature.com/articles/s41597-022-01721-8)
- PneumoniaMNIST++ [`Source`](https://www.nature.com/articles/s41597-022-01721-8)
- TissueMNIST++ [`Source`](https://www.nature.com/articles/s41597-022-01721-8)

### RGB

- CIFAR-10 [`Source`](https://www.cs.toronto.edu/%7Ekriz/cifar.html)
- SVHN [`Source`](https://arxiv.org/abs/1312.6082)

## Repository Information

### Documentation

Full documentation is available here: [`docs/`](docs).

### Dev

See the [Developer](docs/DEVELOPER.md) guidelines for more information.

### Contributing

Contributions of any kind are welcome. Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md]) for details and
the process for submitting pull requests to us.

### Changelog

See the [Changelog](CHANGELOG.md) for more information.

### Security

Thank you for improving the security of the project, please see the [Security Policy](docs/SECURITY.md)
for more information.

## License

This project is licensed under the terms of the `No license` license.
See [LICENSE](LICENSE) for more details.

## References

All the repositories used to generate this code are mentioned in each of the corresponding files. We would like to list them in no particular order:

- [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
- [conditional-GAN](https://github.com/TeeyoHuang/conditional-GAN)
- [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
- [minDiffusion](https://github.com/cloneofsimo/minDiffusion)
- [DenoisingDiffusionProbabilisticModels](https://github.com/DhruvSrikanth/DenoisingDiffusionProbabilisticModels)
- [Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)
- [ddim](https://github.com/ermongroup/ddim)
- [Generative Models](https://github.com/Project-MONAI/GenerativeModels)
- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
    - [SGM Tutorial](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)

## Citation

If you publish work that uses GenerativeZoo, please cite GenerativeZoo as follows:

```bibtex
@misc{GenerativeZoo,
author = {Francisco Caetano},
title = {A model zoo for generative models.},
year = {2024},
}
```
