#!/usr/bin/env bash

cd ../src/generativezoo
python VQGAN_T.py --dataset tinyimagenet --train --embedding_dim 64
poweroff