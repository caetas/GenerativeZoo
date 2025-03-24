#!/usr/bin/env bash

cd src/generativezoo
python VanVAE.py --train --hidden_dims 16 32 64 --n_epochs 20
