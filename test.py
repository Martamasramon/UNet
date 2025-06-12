import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataset.dataset            import MyDataset
from runet_t2w.runetv2          import RUNet
from runet_t2w.test_functions   import visualize_results

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',   type=str,  default='checkpoints_0306_1947_stage_1_best')
parser.add_argument('--img_size',     type=int,  default=128)
parser.add_argument('--drop_first',   type=float,default=0.1)
parser.add_argument('--drop_last',    type=float,default=0.5)
parser.add_argument('--batch_size',   type=int,  default=5)

parser.add_argument('--img_folder',   type=str,  default='/cluster/project7/backup_masramon/IQT/PICAI/T2W/')
parser.add_argument('--is_pretrain',  action='store_true')
parser.add_argument('--finetune',     dest='is_pretrain', action='store_false')
parser.set_defaults(is_pretrain=True)
args, unparsed = parser.parse_known_args()
print('\n',args)

# Set device
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Create dataset
dataset = MyDataset(args.img_folder, img_size=args.img_size, is_pretrain=args.is_pretrain, is_train=False)

print(f"Loading RUNet weights from {args.checkpoint}")
model   = RUNet(args.drop_first, args.drop_last).to(device)
model.load_state_dict(torch.load(f"checkpoints/{args.checkpoint}.pth"))

save_name =  f'{args.checkpoint[12:]}_'
save_name += 'PICAI' if args.is_pretrain else 'HistoMRI'
visualize_results(model, dataset, device, save_name, batch_size=args.batch_size)

# ## EVALUATE
# checkpoint_unet = "checkpoints/perceptual_loss_RUNET_var_blur.pth"
# evaluator_runet = RUNetEvaluation()
# evaluator_runet.evaluate(checkpoint_unet)
