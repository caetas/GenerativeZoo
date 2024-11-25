from models.GAN.AdversarialVAE import *
from data.Dataloaders import *
from utils.util import parse_args_AdversarialVAE
import torch
import wandb

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args_AdversarialVAE()

    size = args.size

    if args.train:
        if not args.no_wandb:
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
                        'sample_and_save_frequency': args.sample_and_save_frequency,
                        'kld_weight': args.kld_weight,
                        },
                        name = 'AdversarialVAE_{}'.format(args.dataset))
        
        train_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, num_workers=args.num_workers, mode='train', size=size, n_patches=args.patches)
        model = AdversarialVAE(input_shape = input_size, input_channels=channels, args=args)
        model.train_model(train_loader)

    elif args.test:
        test_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, mode='val', size=size)
        model = AdversarialVAE(input_shape = input_size, input_channels=channels, args=args)
        model.load_state_dict(torch.load(args.checkpoint))
        model.create_validation_grid(test_loader)
    elif args.sample:
        _, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, mode='val', size=size)
        model = AdversarialVAE(input_shape = input_size, input_channels=channels, args=args)
        model.load_state_dict(torch.load(args.checkpoint))
        model.create_grid()
    elif args.outlier_detection:
        in_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=True, mode='val', size=size)
        out_loader, _, _ = pick_dataset(dataset_name=args.out_dataset, batch_size=args.batch_size, normalize=True, mode='val', size=input_size)
        model = AdversarialVAE(input_shape = input_size, input_channels=channels, args=args)
        if args.checkpoint is not None:
            model.vae.load_state_dict(torch.load(args.checkpoint))
        if args.discriminator_checkpoint is not None:
            model.discriminator.load_state_dict(torch.load(args.discriminator_checkpoint))
        model.eval()
        model.outlier_detection(in_loader, out_loader)
    else:
        Exception("Invalid mode. Set --train, --test or --sample")