print('In script')
      
import torch.optim as optim
import torch
from torch.nn                   import L1Loss
from torch.utils.data           import DataLoader
from torch.optim.lr_scheduler   import ReduceLROnPlateau
import argparse

from dataset.dataset            import MyDataset
# from loss                       import VGGPerceptualLoss, ssim_loss
from runet_t2w.runetv2          import RUNet
from runet_t2w.training_functions_debug import train_evaluate, get_checkpoint_name, CHECKPOINTS_FOLDER


folder = '/cluster/project7/backup_masramon/IQT/'

parser = argparse.ArgumentParser()
# UNet
parser.add_argument('--img_size',       type=int,  default=128)
parser.add_argument('--train_bs',       type=int,  default=8)
parser.add_argument('--test_bs',        type=int,  default=1)
parser.add_argument('--drop_first',     type=float,default=0.1)
parser.add_argument('--drop_last',      type=float,default=0.5)
# Training
parser.add_argument('--n_epochs',       type=int,   default=500)
parser.add_argument('--lr',             type=float, default=1e-4)
parser.add_argument('--factor',         type=float, default=0.5)
parser.add_argument('--patience',       type=int,   default=3)
parser.add_argument('--cooldown',       type=int,   default=1)
# Loss
parser.add_argument('--λ_pixel',        type=float, default=10.0)
parser.add_argument('--λ_perct',        type=float, default=0.)
parser.add_argument('--λ_ssim',         type=float, default=1.0)
parser.add_argument('--stage_delay',    type=int,   default=300)
parser.add_argument('--lr_factor',      type=float, default=0.1)
# Checkpoint
parser.add_argument('--checkpoint',     type=str,   default=None)
parser.add_argument('--save_as',        type=str,   default=None)
# Dataset
parser.add_argument('--finetune',       action='store_true')
parser.add_argument('--use_mask',       action='store_true')
parser.set_defaults(finetune=False) 
parser.set_defaults(use_mask=False) 

args, unparsed = parser.parse_known_args()

def main():
    # Set device
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    print('\nCreating model...')
    model = RUNet(args.drop_first, args.drop_last, args.img_size)
    model = model.to(device)

    # Create dataset & dataloader
    print('Creating datasets...')
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'
    train_dataset = MyDataset(
        folder + data_folder, 
        img_size    = args.img_size, 
        is_finetune = args.finetune, 
        is_train    = True,
        use_mask    = args.use_mask
    )
    test_dataset = MyDataset(
        folder + data_folder, 
        img_size    = args.img_size, 
        is_finetune = args.finetune, 
        is_train    = False,
        use_mask    = args.use_mask
    )

    train_dataloader  = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True,  num_workers=0 )
    test_dataloader   = DataLoader(test_dataset,  batch_size=args.test_bs,  shuffle=False, num_workers=0)

    print('Starting training...')
    losses = {'Pixel': L1Loss(), 'Perceptual': VGGPerceptualLoss(), 'SSIM': ssim_loss}
    losses = {'Pixel': L1Loss(), 'Perceptual': None, 'SSIM': None}

    if args.checkpoint is None:
        # 1. Train only with pixel loss
        print('\n1. Train only with pixel loss...')
        λ_loss = {'Pixel': args.λ_pixel, 'Perceptual': 0, 'SSIM': 0}

        checkpoint = args.save_as if args.save_as is not None else get_checkpoint_name()
        optimizer  = optim.Adam(model.parameters(), lr=args.lr)
        scheduler  = ReduceLROnPlateau(optimizer, 'min', factor=args.factor, patience=args.patience, cooldown=args.cooldown, min_lr=1e-7)
        
        print('before')
        train_evaluate(model, device, train_dataloader, test_dataloader, optimizer, scheduler, args.stage_delay, checkpoint+'_stage_1', losses, λ_loss)
    else:
        checkpoint = args.save_as if args.save_as is not None else args.checkpoint


    

if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()