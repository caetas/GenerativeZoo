from models.GAN.AdversarialVAE import *
from data.Dataloaders import *
from utils.util import parse_args_AdversarialVAE
import torch
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parse_args_AdversarialVAE()

if args.train:
    wandb.init(project='AdversarialVAE',
               config={
                   'dataset': args.dataset,
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'latent_dim': args.latent_dim,
                    'hidden_dims': args.hidden_dims,
                    'lr': args.lr,
                    'gen_weight': args.gen_weight,
                    'recon_weight': args.recon_weight,
                    'sample_and_save_frequency': args.sample_and_save_frequency
                },
                name = 'AdversarialVAE_{}'.format(args.dataset))
    
    train_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, size=32)
    val_loader, _, _ = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, mode='train', size=32)
    model = AdversarialVAE(input_shape = input_size, device = device, input_channels = channels, latent_dim = args.latent_dim, n_epochs = args.n_epochs, hidden_dims = args.hidden_dims, lr = args.lr, batch_size = args.batch_size, gen_weight = args.gen_weight, recon_weight=args.recon_weight, sample_and_save_frequency = args.sample_and_save_frequency, dataset=args.dataset)
    model.train_model(train_loader, val_loader)

elif args.test:
    test_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, mode='val', size=32)
    model = AdversarialVAE(input_shape = input_size, device = device, input_channels = channels, latent_dim = args.latent_dim, n_epochs = args.n_epochs, hidden_dims = args.hidden_dims, lr = args.lr, batch_size = args.batch_size, gen_weight = args.gen_weight, recon_weight=args.recon_weight, sample_and_save_frequency = args.sample_and_save_frequency)
    model.load_state_dict(torch.load(args.checkpoint))
    model.create_validation_grid(test_loader)
elif args.sample:
    _, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, mode='val', size=32)
    model = AdversarialVAE(input_shape = input_size, device = device, input_channels = channels, latent_dim = args.latent_dim, n_epochs = args.n_epochs, hidden_dims = args.hidden_dims, lr = args.lr, batch_size = args.batch_size, gen_weight = args.gen_weight, recon_weight=args.recon_weight, sample_and_save_frequency = args.sample_and_save_frequency)
    model.load_state_dict(torch.load(args.checkpoint))
    model.create_grid()
else:
    Exception("Invalid mode. Set --train, --test or --sample")