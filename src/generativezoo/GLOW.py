from models.Flow.Glow import *
from data.Dataloaders import *
from utils.util import parse_args_Glow
import torch
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args_Glow()
normalize = True

if args.dataset == "mnist" or args.dataset == "fashionmnist":
    size = 32
else:
    size = None

if args.train:
    wandb.init(project='GLOW',
               
               config={
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "n_epochs": args.n_epochs,
                    "dataset": args.dataset,
                    "hidden_channels": args.hidden_channels,
                    "K": args.K,
                    "L": args.L,
                    "actnorm_scale": args.actnorm_scale,
                    "flow_permutation": args.flow_permutation,
                    "flow_coupling": args.flow_coupling,
                    "LU_decomposed": args.LU_decomposed,
                    "learn_top": args.learn_top,
                    "y_condition": args.y_condition,
                    "num_classes": args.num_classes,
                    "n_bits": args.n_bits,  
               },

                name = 'GLOW_{}'.format(args.dataset))
    
    train_loader, input_shape, channels = pick_dataset(args.dataset, batch_size=args.batch_size, normalize=normalize, size=size, num_workers=0)
    model = Glow(image_shape        =   (input_shape,input_shape,channels), 
                 hidden_channels    =   args.hidden_channels, 
                 K                  =   args.K,
                 L                  =   args.L,
                 n_epochs           =   args.n_epochs,
                 actnorm_scale      =   args.actnorm_scale,
                 flow_permutation   =   args.flow_permutation,
                 flow_coupling      =   args.flow_coupling,
                 LU_decomposed      =   args.LU_decomposed,
                 num_classes        =   args.num_classes,
                 learn_top          =   args.learn_top,
                 y_condition        =   args.y_condition,
                 device             =   device,
                 lr                 =   args.lr,
                 n_bits             =   args.n_bits,
                 dataset            =   args.dataset)
    model.train_model(train_loader)

elif args.sample:
    _, input_shape, channels = pick_dataset(args.dataset, batch_size=args.batch_size, normalize=normalize, size=size, num_workers=0)
    model = Glow(image_shape        =   (input_shape,input_shape,channels), 
                 hidden_channels    =   args.hidden_channels, 
                 K                  =   args.K,
                 L                  =   args.L,
                 actnorm_scale      =   args.actnorm_scale,
                 flow_permutation   =   args.flow_permutation,
                 flow_coupling      =   args.flow_coupling,
                 LU_decomposed      =   args.LU_decomposed,
                 num_classes        =   args.num_classes,
                 learn_top          =   args.learn_top,
                 y_condition        =   args.y_condition,
                 device             =   device,
                 n_bits             =   args.n_bits)
    model.load_state_dict(torch.load(args.checkpoint))
    model.sample(train=False)

elif args.outlier_detection:
    in_loader, input_shape, channels = pick_dataset(args.dataset, batch_size=args.batch_size, normalize=normalize, size=size, num_workers=0, mode='val')
    out_loader, _, _ = pick_dataset(args.out_dataset, batch_size=args.batch_size, normalize=normalize, size=input_shape, num_workers=0, mode='val')
    model = Glow(image_shape        =   (input_shape,input_shape,channels), 
                 hidden_channels    =   args.hidden_channels, 
                 K                  =   args.K,
                 L                  =   args.L,
                 actnorm_scale      =   args.actnorm_scale,
                 flow_permutation   =   args.flow_permutation,
                 flow_coupling      =   args.flow_coupling,
                 LU_decomposed      =   args.LU_decomposed,
                 num_classes        =   args.num_classes,
                 learn_top          =   args.learn_top,
                 y_condition        =   args.y_condition,
                 device             =   device,
                 n_bits             =   args.n_bits)
    model.load_state_dict(torch.load(args.checkpoint))
    model.outlier_detection(in_loader, out_loader)

else:
    raise ValueError("Invalid mode. Please specify train or sample")
