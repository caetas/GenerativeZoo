from models.AR.VQGAN_Transformer import *
from data.Dataloaders import *
from utils.util import parse_args_VQGAN_Transformer
import wandb

args = parse_args_VQGAN_Transformer()

if args.dataset == 'mnist':
    size = 32
else:
    size = None

if args.train:
    train_loader_a, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=False, size=size)
    train_loader_b, _, _ = pick_dataset(args.dataset, 'train', args.batch_size//4, normalize=False, size=size)
    wandb.init(project='VQGAN_Transformer',
               config={
                   'dataset': args.dataset,
                   'batch_size': args.batch_size,
                   'n_epochs': args.n_epochs,
                   'n_epochs_t': args.n_epochs_t,
                   'lr': args.lr,
                   'lr_t': args.lr_t,
                   'lr_d': args.lr_d,
                   'num_res_layers': args.num_res_layers,
                   'downsample_parameters': args.downsample_parameters,
                   'upsample_parameters': args.upsample_parameters,
                   'num_channels': args.num_channels,
                   'num_res_channels': args.num_res_channels,
                   'num_embeddings': args.num_embeddings,
                   'embedding_dim': args.embedding_dim,
                   'attn_layers_dim': args.attn_layers_dim,
                   'attn_layers_depth': args.attn_layers_depth,
                   'attn_layers_heads': args.attn_layers_heads,
                   'num_layers_d': args.num_layers_d,
                   'num_channels_d': args.num_channels_d,
                   'adv_weight': args.adv_weight,
                   'perceptual_weight': args.perceptual_weight,
                 },

                name='VQGAN_Transformer_{}'.format(args.dataset))
    model = VQGANTransformer(args, channels=channels, img_size=input_size)
    model.train_model(args, train_loader_a, train_loader_b)
    wandb.finish()

elif args.sample:
    _, input_size, channels = pick_dataset(args.dataset, 'train', 1, normalize=False, size=size)
    model = VQGANTransformer(args, channels=channels, img_size=input_size)
    model.load_checkpoint(args.checkpoint, args.checkpoint_t)
    model.sample(16, train=False)