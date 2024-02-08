# GenerativeZoo

[![Python](https://img.shields.io/badge/python-3.9+-informational.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=black)](https://pycqa.github.io/isort)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://mkdocstrings.github.io)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![mlflow](https://img.shields.io/badge/tracking-mlflow-blue)](https://mlflow.org)
[![dvc](https://img.shields.io/badge/data-dvc-9cf)](https://dvc.org)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![pytest](https://img.shields.io/badge/pytest-enabled-brightgreen)](https://github.com/pytest-dev/pytest)
[![conventional-commits](https://img.shields.io/badge/conventional%20commits-1.0.0-yellow)](https://github.com/commitizen-tools/commitizen)

A short description of the project. No quotes.

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

And then setup all virtualenv using make file recipe

    (python3.9) $ make setup-all

## Documentation

Full documentation is available here: [`docs/`](docs).

## Models

### Implemented Models

The listed models are already implemented and fully integrated in the model zoo.

#### VAEs
- Vanilla VAE
- Conditional VAE

#### GANs
- Adversarial VAE
- Vanilla GAN
- Conditional GAN
- CycleGAN

#### DDPMs
- Unconditional DDPM
- Conditional DDPM
- Diffusion AE

#### SGMs
- Unconditional SGM

### Future Models

These models are currently under development and will be added to the repository in the future.

#### VAEs
- [ ] Hierarchical VAE
- [ ] VQ-VAE

#### GANs
- [ ] VQ-GAN

#### SGMs
- [ ] NCSN
- [ ] NCSN++

#### Autoregressive
- [ ] VQ-VAE + Transformer
- [ ] VQ-VAE + Mamba

#### Flow-Based Models
- [ ] Flow++

## Dev

See the [Developer](docs/DEVELOPER.md) guidelines for more information.

## Contributing

Contributions of any kind are welcome. Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md]) for details and
the process for submitting pull requests to us.

## Changelog

See the [Changelog](CHANGELOG.md) for more information.

## Security

Thank you for improving the security of the project, please see the [Security Policy](docs/SECURITY.md)
for more information.

## License

This project is licensed under the terms of the `No license` license.
See [LICENSE](LICENSE) for more details.

## Citation

If you publish work that uses GenerativeZoo, please cite GenerativeZoo as follows:

```bibtex
@misc{GenerativeZoo,
  author = {Francisco Caetano},
  title = {A model zoo for generative models.},
  year = {2024},
}
```
