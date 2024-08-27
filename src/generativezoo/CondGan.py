from models.GAN.ConditionalGAN import *
from data.Dataloaders import *
from utils.util import parse_args_CondGAN
import torch
import wandb

if __name__ == '__main__':

    args = parse_args_CondGAN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = None

    if args.train:
        if not args.no_wandb:
            wandb.init(project='ConditionalGAN',
                    config={
                        'dataset': args.dataset,
                        'batch_size': args.batch_size,
                        'n_epochs': args.n_epochs,
                        'latent_dim': args.latent_dim,
                        'd': args.d,
                        'lr': args.lr,
                        'beta1': args.beta1,
                        'beta2': args.beta2,
                        'sample_and_save_freq': args.sample_and_save_freq
                    },
                    name = 'ConditionalGAN_{}'.format(args.dataset))
        train_dataloader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size = size, num_workers=args.num_workers)
        model = ConditionalGAN(img_size=input_size, channels=channels, args=args)
        model.train_model(train_dataloader)

    elif args.sample:
        _, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=1, normalize = True, size = size)
        model = Generator(n_classes = args.n_classes, latent_dim = args.latent_dim, channels = channels, d = args.d).to(device)
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        model.sample(n_samples = args.n_samples, device = device)
    else:
        raise Exception('Please specify either --train or --sample')
