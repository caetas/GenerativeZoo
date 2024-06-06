<p align="center">
  <img src="imgs/logo_tasti_light.png" width="70%" alt='Tasti Project'>
</p>

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

**This work is part of the Xecs TASTI project, nr. 2022005.**

## Prerequisites

You will need:

- `python` (see `pyproject.toml` for full version)
- `Git`
- `Make`
- a `.secrets` file with the required secrets and credentials
- load environment variables from `.env`
- `CUDA >= 12.1`

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
    .venv-dev/Scripts/Activate.ps1
    python -m pip install --upgrade pip setuptools
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -r requirements/requirements-win.txt


To run the code please remember to always activate both environments:

    conda activate python3.9
    .venv-dev/Scripts/Activate.ps1

## Models

### Implemented Models

The listed models are already implemented and fully integrated in the model zoo.

#### VAEs

- Vanilla VAE [`Paper`](https://arxiv.org/abs/1312.6114)|[`Code`](src/generativezoo/models/VAE/VanillaVAE.py)|[`Script`](src/generativezoo/VanVAE.py)|[`Documentation`](docs/VanillaVAE.md)
- Conditional VAE [`Paper`](https://openreview.net/forum?id=rJWXGDWd-H)|[`Code`](src/generativezoo/models/VAE/ConditionalVAE.py)|[`Script`](src/generativezoo/CondVAE.py)|[`Documentation`](docs/ConditionalVAE.md)
- Hierarchical VAE [`Paper`](https://arxiv.org/abs/2007.03898)|[`Code`](src/generativezoo/models/VAE/HierarchicalVAE.py)|[`Script`](src/generativezoo/HVAE.py)|[`Documentation`](docs/HierarchicalVAE.md)

#### GANs

- Adversarial VAE [`Paper`](https://arxiv.org/abs/1511.05644)|[`Code`](src/generativezoo/models/GAN/AdversarialVAE.py)|[`Script`](src/generativezoo/AdvVAE.py)|[`Documentation`](docs/AdversarialVAE.md)
- Vanilla GAN [`Paper`](https://arxiv.org/abs/1511.06434)|[`Code`](src/generativezoo/models/GAN/VanillaGAN.py)|[`Script`](src/generativezoo/VanGAN.py)|[`Documentation`](docs/VanillaGAN.md)
- Conditional GAN [`Paper`](https://arxiv.org/abs/1411.1784)|[`Code`](src/generativezoo/models/GAN/ConditionalGAN.py)|[`Script`](src/generativezoo/CondGAN.py)|[`Documentation`](docs/ConditionalGAN.md)
- CycleGAN [`Paper`](https://arxiv.org/abs/1703.10593)|[`Code`](src/generativezoo/models/GAN/CycleGAN.py)|[`Script`](src/generativezoo/CycGAN.py)|[`Documentation`](docs/CycleGAN.md)
- Prescribed GAN [`Paper`](https://arxiv.org/abs/1910.04302)|[`Code`](src/generativezoo/models/GAN/PrescribedGAN.py)|[`Script`](src/generativezoo/PresGAN.py)|[`Documentation`](docs/PrescribedGAN.md)
- Wasserstein GAN with Gradient Penalty [`Paper`](https://arxiv.org/abs/1704.00028)|[`Code`](src/generativezoo/models/GAN/WGAN.py)|[`Script`](src/generativezoo/WGAN.py)|[`Documentation`](docs/WassersteinGAN.md)

#### DDPMs

- Vanilla DDPM [`Paper`](https://arxiv.org/abs/2006.11239)|[`Code`](src/generativezoo/models/DDPM/VanillaDDPM.py)|[`Script`](src/generativezoo/VanDDPM.py)|[`Documentation`](docs/VanillaDDPM.md)
- Conditional DDPM [`Paper`](https://arxiv.org/abs/2207.12598)|[`Code`](src/generativezoo/models/DDPM/ConditionalDDPM.py)|[`Script`](src/generativezoo/CondDDPM.py)|[`Documentation`](docs/ConditionalDDPM.md)
- Diffusion AE [`Paper`](https://arxiv.org/abs/2111.15640)|[`Code`](src/generativezoo/models/DDPM/MONAI_DiffAE.py)|[`Script`](src/generativezoo/DAE.py)|[`Documentation`](docs/DiffusionAE.md)

#### SGMs

- Vanilla SGM [`Paper`](https://arxiv.org/abs/2006.09011)|[`Code`](src/generativezoo/models/SGM/VanillaSGM.py)|[`Script`](src/generativezoo/VanSGM.py)|[`Documentation`](docs/VanillaSGM.md)
- NCSNv2 [`Paper`](https://arxiv.org/abs/2006.09011)|[`Code`](src/generativezoo/models/SGM/NCSNv2.py)|[`Script`](src/generativezoo/NCSNv2.py)|[`Documentation`](docs/NCSNv2.md)

#### Autoregressive

- VQ-VAE + Transformer [`Paper`](https://arxiv.org/abs/1711.00937)|[`Code`](src/generativezoo/models/AR/VQVAE_Transformer.py)|[`Script`](src/generativezoo/VQVAE_T.py)|[`Documentation`](docs/VQVAE_Transformer.md)
- VQ-GAN + Transformer [`Paper`](https://arxiv.org/abs/2012.09841)|[`Code`](src/generativezoo/models/AR/VQGAN_Transformer.py)|[`Script`](src/generativezoo/VQGAN_T.py)|[`Documentation`](docs/VQGAN_Transformer.md)
- PixelCNN [`Paper`](https://arxiv.org/abs/1606.05328)|[`Code`](src/generativezoo/models/AR/PixelCNN.py)|[`Script`](src/generativezoo/P-CNN.py)|[`Documentation`](docs/PixelCNN.md)

#### Flow

- Glow [`Paper`](https://arxiv.org/abs/1807.03039)|[`Code`](src\generativezoo\models\Flow\Glow.py)|[`Script`](src/generativezoo/GLOW.py)|[`Documentation`](docs/Glow.md)
- Flow++ [`Paper`](https://arxiv.org/abs/1902.00275)|[`Code`](src\generativezoo\models\Flow\FlowPlusPlus.py)|[`Script`](src\generativezoo\FlowPP.py)|[`Documentation`](docs/FlowPlusPlus.md)

### Future Models

These models are currently under development and will be added to the repository in the future.

#### Autoregressive

- [ ] VQ-VAE + Mamba

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
- TinyImageNet [`Source`](https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf) **MANUAL DOWNLOAD REQUIRED** [`Link`](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200)

## Tracking

The code examples are setup to use [Weights & Biases](https://wandb.ai/home) as a tool to track your training runs. Please refer to the [`full documentation`](https://docs.wandb.ai/quickstart) if required or follow the following steps:

1. Create an account in [Weights & Biases](https://wandb.ai/home)
2. **If you have installed the requirements you can skip this step**. If not, activate the conda environment and the virtualenv and run:

    ```bash
    pip install wandb
    ```
3. Run the following command and insert you [`API key`](https://wandb.ai/authorize) when prompted:

    ```bash
    wandb login
    ```

## Repository Information

### Documentation

Full documentation is available here: [`docs/`](docs).

### Dev

See the [Developer](docs/DEVELOPER.md) guidelines for more information.

### Contributing

Contributions of any kind are welcome. Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md]) for details and
the process for submitting pull requests to us.

**Please read [MODELRULES.md](docs/MODELRULES.md) for details on how you should build your models for this repository.**

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
- [wgan-gp](https://github.com/EmilienDupont/wgan-gp)
- [PresGANs](https://github.com/adjidieng/PresGANs/)
- [minDiffusion](https://github.com/cloneofsimo/minDiffusion)
- [DenoisingDiffusionProbabilisticModels](https://github.com/DhruvSrikanth/DenoisingDiffusionProbabilisticModels)
- [Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)
- [ddim](https://github.com/ermongroup/ddim)
- [Generative Models](https://github.com/Project-MONAI/GenerativeModels)
- [Glow-PyTorch](https://github.com/y0ast/Glow-PyTorch)
- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
    - [SGM Tutorial](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)

## Citation

If you publish work that uses GenerativeZoo, please cite GenerativeZoo as follows:

```bibtex
@misc{GenerativeZoo,
author = {Francisco Caetano},
title = {A collection of generative algorithms and techniques implemented in Python.},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/caetas/GenerativeZoo}},
year = {2024},
}
```
