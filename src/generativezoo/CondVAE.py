from models.VAE.ConditionalVAE import *
from data.Dataloaders import *
import torch
from utils.util import parse_args_ConditionalVAE
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args_ConditionalVAE()

if args.train:
    # train dataloader
    train_loader, in_shape, in_channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, size=32)
    wandb.init(project='CVAE',
                
                config={
                    'dataset': args.dataset,
                    'batch_size': args.batch_size,
                    'n_epochs': args.n_epochs,
                    'lr': args.lr,
                    'latent_dim': args.latent_dim,
                    'hidden_dims': args.hidden_dims,
                    'input_size': in_shape,
                    'channels': in_channels,
                    'n_classes': args.n_classes
                },

                name = 'CVAE_{}'.format(args.dataset))
    # create model
    model = ConditionalVAE(input_shape=in_shape, input_channels=in_channels, num_classes=args.n_classes, latent_dim=args.latent_dim, batch_size=args.batch_size, device=device, hidden_dims=args.hidden_dims, lr=args.lr, sample_and_save_freq=args.sample_and_save_freq, dataset=args.dataset)
    # train model
    model.train_model(train_loader, args.n_epochs)

elif args.sample:
    _, in_shape, in_channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, size=32)
    model = ConditionalVAE(input_shape=in_shape, input_channels=in_channels, latent_dim=args.latent_dim, batch_size=args.batch_size, device=device, hidden_dims=args.hidden_dims, lr=args.lr)
    model.load_state_dict(torch.load(args.checkpoint))
    model.create_grid(title="Sample", train = False)
else:
    raise ValueError("Invalid mode. Please specify train or sample")