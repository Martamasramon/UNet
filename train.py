import torch.optim as optim
import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import ReduceLROnPlateau

from dataset.dataset    import MyDataset
from dataset.transforms import create_transforms
from loss               import VGGPerceptualLoss
from runet.runet        import RUNet
from runet.training_functions import train_evaluate
from utils.formatter    import get_checkpoint_name

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_size',       type=int,  default=128)
parser.add_argument('--train_bs',       type=int,  default=16)
parser.add_argument('--test_bs',        type=int,  default=1)
parser.add_argument('--drop_first',     type=float,default=0.1)
parser.add_argument('--drop_last',      type=float,default=0.5)

parser.add_argument('--n_epochs',   type=int,   default=200)
parser.add_argument('--lr',         type=float, default=1e-5)
parser.add_argument('--factor',     type=float, default=0.5)
parser.add_argument('--patience',   type=int,   default=4)
parser.add_argument('--cooldown',   type=int,   default=2)

parser.add_argument('--img_folder',   type=str,  default='/cluster/project7/backup_masramon/IQT/T2W/')

args, unparsed = parser.parse_known_args()
print('\n',args)

# Create model
print('\nCreating model...')
model = RUNet(args.drop_first, args.drop_last)
model = model.cuda()

# Create dataset & dataloader
print('Creating datasets...')
train_transforms, test_transforms = create_transforms(args.img_size)
train_dataset   = MyDataset(args.img_folder, train_transforms, is_pretrain=True, is_train=True)
test_dataset    = MyDataset(args.img_folder, test_transforms,  is_pretrain=True, is_train=False)

train_dataloader  = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True,  num_workers=8)
test_dataloader   = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False, num_workers=0)

pixel_fn = L1Loss()  # or MSELoss if preferred

# Run training - Train only with pixel loss
print('Starting training...')
checkpoint = get_checkpoint_name()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.factor, patience=args.patience, cooldown=args.cooldown, min_lr=1e-7)
train_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, args.n_epochs, checkpoint, pixel_fn)