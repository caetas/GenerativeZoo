from models.SGM.NCSNv2 import *
from utils.util import parse_args_NCSNv2
import torch
from data.Dataloaders import *

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parse_args_NCSNv2()

if args.train:
    train_loader, input_size, channels = pick_dataset(dataset_name=args.dataset, batch_size=args.batch_size, normalize=False, size=32)
    print(input_size, channels)
    model = NCSNv2(input_size, channels, args)
    model.train_model(train_loader, args)