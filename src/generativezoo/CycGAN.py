from models.GANs.CycleGAN import *
from data.CycleGAN_Dataloaders import *
from config import data_raw_dir, models_dir
import torch
import wandb
import subprocess
from utils.util import parse_args_CycleGAN

args = parse_args_CycleGAN()

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.train:

    subprocess.run(['wandb', 'login'])
    wandb.init(project='CycleGAN',
               config={
                   'dataset': args.dataset,
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'lr': args.lr,
                   'decay': args.decay,
                   'input_size': args.input_size,
                   'in_channels': args.in_channels,
                   'out_channels': args.out_channels,
                   'sample_and_save_freq': args.sample_and_save_freq
               },
               name = 'CycleGAN_{}'.format(args.dataset))
    
    train_dataloader_A = get_horse2zebra_dataloader(data_raw_dir, args.dataset, args.batch_size, True, 'A', args.input_size)
    train_dataloader_B = get_horse2zebra_dataloader(data_raw_dir, args.dataset, args.batch_size, True, 'B', args.input_size)
    test_dataloader_A = get_horse2zebra_dataloader(data_raw_dir, args.dataset, args.batch_size, False, 'A', args.input_size)
    test_dataloader_B = get_horse2zebra_dataloader(data_raw_dir, args.dataset, args.batch_size, False, 'B', args.input_size)

    train(train_dataloader_A, train_dataloader_B, test_dataloader_A, test_dataloader_B, args.n_epochs, args.lr, args.decay, args.in_channels, args.out_channels, device, args.sample_and_save_freq, args.dataset)

elif args.test:
    test_dataloader_A = get_horse2zebra_dataloader(data_raw_dir, args.dataset, args.batch_size, False, 'A', args.input_size)
    test_dataloader_B = get_horse2zebra_dataloader(data_raw_dir, args.dataset, args.batch_size, False, 'B', args.input_size)

    test(args.checkpoint_A, args.checkpoint_B, test_dataloader_A, test_dataloader_B, args.in_channels, args.out_channels, device)

else:
    raise Exception('Please specify either --train or --test')