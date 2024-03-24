#!/usr/bin/env bash

cd ../src/generativezoo
python VanGAN.py --dataset tinyimagenet --d 64 --latent_dim 1024 --batch_size 512 --n_epochs 200 --train
python PresGAN.py --dataset tinyimagenet --restrict_sigma 1 --sigma_min 1e-3 --sigma_max 0.3 --lambda 5e-4 --n_epochs 200 --train --batch_size 512 --nz 1024
poweroff