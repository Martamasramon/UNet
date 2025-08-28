import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

from arguments          import args
from dataset.dataset    import MyDataset
from torch.utils.data   import DataLoader

from runet_t2w.runetv2          import RUNet
from runet_t2w.test_functions   import visualize_results, evaluate_results
from runet_t2w.training_functions import CHECKPOINTS_FOLDER

folder = '/cluster/project7/backup_masramon/IQT/'

args, unparsed = parser.parse_known_args()


def main():
    # Set device
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    data_folder = 'HistoMRI' if args.finetune else 'PICAI'
    dataset = MyDataset(
        folder + data_folder, 
        data_type   = 'val',
        img_size    = args.img_size, 
        is_finetune = args.finetune, 
        use_mask    = args.use_mask
    )

    # Load model
    print(f"Loading RUNet weights from {args.checkpoint}")
    model   = RUNet(args.drop_first, args.drop_last, args.img_size).to(device)
    model.load_state_dict(torch.load(f"{CHECKPOINTS_FOLDER}{args.checkpoint}.pth"))

    print(f"Visualising results...")
    save_name = args.checkpoint+'_HistoMRI' if args.finetune else args.checkpoint+'_PICAI' 
    save_name = save_name+'_mask' if args.use_mask else save_name
    visualize_results(model, dataset, device, save_name, batch_size=args.test_bs)

    print(f"Evaluating results...")
    dataloader = DataLoader(dataset,  batch_size=args.test_bs,  shuffle=False)
    evaluate_results(model, dataloader, device)
    
if __name__ == '__main__':
    print('Parameters:')
    for key, value in vars(args).items():
        print(f'- {key}: {value}')
    print('')
    
    main()
