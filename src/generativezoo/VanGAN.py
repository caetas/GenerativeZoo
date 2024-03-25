from models.GAN.VanillaGAN import *
from data.Dataloaders import *
from utils.util import parse_args_VanillaGAN
import torch
import wandb

args = parse_args_VanillaGAN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.train:
    wandb.init(project='VanillaGAN',
               config={
                   'dataset': args.dataset,
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'latent_dim': args.latent_dim,
                   'd': args.d,
                   'lrg': args.lrg,
                   'lrd': args.lrd,
                   'beta1': args.beta1,
                   'beta2': args.beta2,
                   'sample_and_save_freq': args.sample_and_save_freq
               },
               name = 'VanillaGAN_{}'.format(args.dataset))
    train_dataloader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size = 32)
    model = VanillaGAN(n_epochs = args.n_epochs, device = device, latent_dim = args.latent_dim, d = args.d, channels = channels, lrg = args.lrg, lrd=args.lrd, beta1 = args.beta1, beta2 = args.beta2, img_size = input_size, sample_and_save_freq = args.sample_and_save_freq, dataset=args.dataset)
    model.train_model(train_dataloader)

elif args.sample:
    _, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=1, normalize = True, size = 32)
    model = Generator(latent_dim = args.latent_dim, channels = channels, d=args.d, imgSize=input_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    model.sample(n_samples = args.n_samples, device = device)

elif args.outlier_detection:
    in_loader, input_size, channels = pick_dataset(dataset_name = args.dataset, batch_size=args.batch_size, normalize = True, size = 32, mode='val')
    out_loader, _, _ = pick_dataset(dataset_name = args.out_dataset, batch_size=args.batch_size, normalize = True, size = 32, mode='val')
    model = Discriminator(channels=channels, d=args.d).to(device)
    model.load_state_dict(torch.load(args.discriminator_checkpoint))
    model.outlier_detection(in_loader, out_loader, display=True, device=device)

else:
    raise Exception('Please specify either --train, --sample or --outlier_detection. For more information use --help.')