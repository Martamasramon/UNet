import argparse
parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--img_size',       type=int,  default=64)
parser.add_argument('--train_bs',       type=int,  default=16)
parser.add_argument('--test_bs',        type=int,  default=1)  # 15 for test
# UNet
parser.add_argument('--drop_first',     type=float,default=0.1)
parser.add_argument('--drop_last',      type=float,default=0.5)
# Training
parser.add_argument('--factor',         type=float, default=0.1)
parser.add_argument('--patience',       type=int,   default=3)
parser.add_argument('--cooldown',       type=int,   default=1)

parser.add_argument('--n_epochs',       type=int,   default=150)
parser.add_argument('--lr',             type=float, default=1e-4)
parser.add_argument('--n_epochs_2',     type=int,   default=150)
parser.add_argument('--lr_2',           type=float, default=0.1)

parser.add_argument('--λ_pixel',        type=float, default=10.0)
parser.add_argument('--λ_perct',        type=float, default=0.01)
parser.add_argument('--λ_ssim',         type=float, default=1.0)
# Checkpoint
parser.add_argument('--checkpoint',     type=str,   default=None) # 'default_64_stage_1_best' for test
parser.add_argument('--save_as',        type=str,   default=None)
# Dataset
parser.add_argument('--finetune',       action='store_true')
parser.add_argument('--masked',         action='store_true')
parser.set_defaults(finetune=False) 
parser.set_defaults(masked=False) 

args, unparsed = parser.parse_known_args()