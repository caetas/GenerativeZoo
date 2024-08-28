from models.VAE.VanillaVAE import *
from data.Dataloaders import *
import torch
from utils.util import parse_args_VanillaVAE
import wandb

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args_VanillaVAE()

    size = None

    if args.train:
        # train dataloader
        train_loader, in_shape, in_channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, size = size, num_workers=args.num_workers)
        if not args.no_wandb:
            wandb.init(project='VAE',
                        
                        config={
                            'dataset': args.dataset,
                            'batch_size': args.batch_size,
                            'n_epochs': args.n_epochs,
                            'lr': args.lr,
                            'latent_dim': args.latent_dim,
                            'hidden_dims': args.hidden_dims,
                            'input_size': in_shape,
                            'channels': in_channels,
                            'loss_type': args.loss_type,
                            'kld_weight': args.kld_weight
                        },

                        name = 'VAE_{}'.format(args.dataset))
        # create model
        model = VanillaVAE(input_shape=in_shape, input_channels=in_channels,args=args)
        # train model
        model.train_model(train_loader, args.n_epochs)

    elif args.sample:
        _, in_shape, in_channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, size = size)
        model = VanillaVAE(input_shape=in_shape, input_channels=in_channels,args=args)
        model.load_state_dict(torch.load(args.checkpoint))
        model.sample(title="Sample", train = False)

    elif args.outlier_detection:
        in_loader, in_shape, in_channels = pick_dataset(args.dataset, batch_size = args.batch_size, normalize=True, size = size, mode='val')
        out_loader, _, _ = pick_dataset(args.out_dataset, batch_size = args.batch_size, normalize=True, size = in_shape, mode='val')
        model = VanillaVAE(input_shape=in_shape, input_channels=in_channels,args=args)
        model.load_state_dict(torch.load(args.checkpoint))
        model.outlier_detection(in_loader, out_loader)
    else:
        raise ValueError("Invalid mode. Please specify train or sample")