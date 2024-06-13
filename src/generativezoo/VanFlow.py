from models.Flow.VanillaFlow import VanillaFlow
from utils.util import parse_args_VanillaFlow
from data.Dataloaders import *

args = parse_args_VanillaFlow()
in_loader, img_size, channels = pick_dataset(args.dataset, 'train', args.batch_size)
model = VanillaFlow(img_size, channels, args)
print(model.flows)
model.train_model(in_loader, args)