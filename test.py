import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataset                    import MyDataset
from torch.utils.data           import DataLoader
from runet_t2w.runetv2          import RUNet
from runet_t2w.test_functions   import visualize_results, evaluate_results
from runet_t2w.training_functions import CHECKPOINTS_FOLDER

folder = '/cluster/project7/backup_masramon/IQT/'

parser = argparse.ArgumentParser()
parser.add_argument('--img_size',     type=int,  default=64)
parser.add_argument('--drop_first',   type=float,default=0.1)
parser.add_argument('--drop_last',    type=float,default=0.5)
parser.add_argument('--batch_size',   type=int,  default=5)
# Load checkpoint
parser.add_argument('--checkpoint',   type=str,  default='checkpoints_0306_1947_stage_1_best')
# Dataset
parser.add_argument('--finetune',     action='store_true')
parser.add_argument('--use_mask',     action='store_true')
parser.set_defaults(finetune=False) 
parser.set_defaults(use_mask=False) 

args, unparsed = parser.parse_known_args()
print('\n',args)

# Set device
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset
data_folder = 'HistoMRI' if args.finetune else 'PICAI'
dataset = MyDataset(
    folder + data_folder, 
    img_size    = args.img_size, 
    is_finetune = args.finetune, 
    is_train    = False,
    use_mask    = args.use_mask
)

# Load model
print(f"Loading RUNet weights from {args.checkpoint}")
model   = RUNet(args.drop_first, args.drop_last, args.img_size).to(device)
model.load_state_dict(torch.load(f"{CHECKPOINTS_FOLDER}{args.checkpoint}.pth"))

save_name = args.checkpoint+'_HistoMRI' if args.finetune else args.checkpoint+'_PICAI' 
visualize_results(model, dataset, device, save_name, batch_size=args.batch_size)

# EVALUATE
dataloader = DataLoader(dataset,  batch_size=args.batch_size,  shuffle=False)
evaluate_results(model, dataloader, device, args.batch_size)
