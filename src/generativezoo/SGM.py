from models.SGM.VanillaSGM import train, sample, outlier_detection
from data.Dataloaders import *
from utils.util import parse_args_VanillaSGM
from config import models_dir
import torch
import os
import mlflow

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args_VanillaSGM()
normalize = False

if args.train:
    mlflow.set_experiment('VanillaSGM')
    mlflow.start_run()
    # set run name
    mlflow.set_tag('mlflow.runName', model_name='VanillaSGM_' + args.dataset)
    mlflow.log_param('dataset', args.dataset)
    dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize)
    train(dataloader, device, n_epochs=args.n_epochs, input_size=input_size, in_channels=channels, model_name='VanillaSGM_' + args.dataset, lr=args.lr, sigma = args.sigma)
elif args.sample:
    _, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize)
    sample(args.checkpoint, sampler_type=args.sampler_type, device = device, num_samples=args.num_samples, num_steps=args.num_steps, channels=channels, input_size=input_size)
elif args.outlier_detection:
    dataloader_a, input_size_a, channels_a = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize)
    dataloader_b, input_size_b, channels_b = pick_dataset(args.out_dataset, 'val', args.batch_size, normalize=normalize)
    outlier_detection(args.checkpoint, dataloader_a, dataloader_b, device, args.sigma, input_size=input_size_a, channels=channels_a)
else:
    raise ValueError('Please specify at least one of the following: train, sample, outlier_detection')

