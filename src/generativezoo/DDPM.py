import torch
from data.Dataloaders import *
from models.DDPM.DDPM import *
from utils.util import parse_args_DDPM
import wandb

if __name__ == '__main__':

     device = "cuda" if torch.cuda.is_available() else "cpu"
     args = parse_args_DDPM()
     normalize = True

     if args.train:
          dataloader, input_size, channels = pick_dataset(args.dataset, 'train', args.batch_size, normalize=normalize, size=args.size, num_workers=args.num_workers)
          model = DDPM(args, channels=channels, image_size=input_size)
          model.train_model(dataloader)
          wandb.finish()

     elif args.sample:
          _, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize, size=args.size)
          model = DDPM(args, channels=channels, image_size=input_size)
          model.model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
          model.sample(args.num_samples)

     elif args.inpaint:
          dataloader, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize, size=args.size)
          model = DDPM(args, channels=channels, image_size=input_size)
          model.model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
          model.inpaint(dataloader)

     elif args.outlier_detection:
          dataloader_a, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize, size=args.size)
          model = DDPM(args, channels=channels, image_size=input_size)
          model.model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
          dataloader_b, input_size_b, channels_b = pick_dataset(args.out_dataset, 'val', args.batch_size, normalize=normalize, good = False, size=input_size)
          model.outlier_detection(dataloader_a,dataloader_b, args.dataset, args.out_dataset)

     elif args.fid:
          _, input_size, channels = pick_dataset(args.dataset, 'val', args.batch_size, normalize=normalize, size=args.size)
          model = DDPM(args, channels=channels, image_size=input_size)
          if args.checkpoint is not None:
               model.model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
          model.fid_sample(args.batch_size)

     else:
          raise ValueError('Please specify at least one of the following: train, sample, outlier_detection')