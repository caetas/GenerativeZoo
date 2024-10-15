#!/usr/bin/env bash

cd ../src/generativezoo
python AdvVAE.py --train --dataset tinyimagenet --batch_size 512 --hidden_dims 64 128 256 512 --latent_dim 1024 --n_epochs 140 --lr 5e-4 --gen_weight 1e-3 --recon_weight 1e-3 --sample_and_save_freq 10
cd ../../models/AdversarialVAE
mkdir tiny_cvpr_run3
mv Discriminator_tinyimagenet* tiny_cvpr_run3
poweroff
