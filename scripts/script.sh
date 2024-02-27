#!/usr/bin/env bash

cd ../src/generativezoo
python AdvVAE.py --train --dataset cityscapes --latent_dim 2048 --hidden_dims 64 128 256 --n_epochs 200 --lr 1e-4 --gen_weight 0.0005 --recon_weight 0.0005 --batch_size 32
poweroff