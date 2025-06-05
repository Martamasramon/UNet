import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import utils

from dataset.transforms import create_transforms
from dataset.dataset    import MyDataset
from runet.runetv2 import RUNet

img_folder   = '/cluster/project7/backup_masramon/IQT/PICAI/T2W/'

class RUNetVisualizer:
    def __init__(self, drop_first, drop_last, img_size=128, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        train_transforms, test_transforms = create_transforms(img_size)
        self.train_dataloader   = MyDataset(img_folder, train_transforms, is_pretrain=True, is_train=True)
        self.test_dataloader    = MyDataset(img_folder, test_transforms,  is_pretrain=True, is_train=False)
        self.device=device
        self.model = RUNet(drop_first, drop_last).to(device)
        self.model.eval()

    def format_image(self, img):
        return np.squeeze((img).cpu().numpy())

    def visualize_runet(self, checkpoint, eval_test=True, batch_size=5,seed=1):
        dataloader = self.test_dataloader if eval_test else self.train_dataloader

        print(f"Loading RUNet weights from {checkpoint}")
        self.model.load_state_dict(torch.load(checkpoint))

        fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(3*2,3*batch_size))
        axes[0,0].set_title('Original (input)')
        axes[0,1].set_title('Reconstructed (output)')
        np.random.seed(seed)
        indices = np.random.choice(np.arange(len(dataloader)),batch_size,replace=False)

        with torch.no_grad():
            for i,ind in enumerate(indices):
                sample = dataloader[ind]
                img       = sample["image"].unsqueeze(0).float().to(self.device)            
                img_super = self.model(img)

                im0 =axes[i,0].imshow(self.format_image(img),        cmap='gray', vmin=0, vmax=1)
                axes[i,0].axis('off')
                plt.colorbar(im0, ax=axes[i,0])
                im1 =axes[i,1].imshow(self.format_image(img_super),  cmap='gray', vmin=0, vmax=1)
                axes[i,1].axis('off')
                plt.colorbar(im1, ax=axes[i,1])
                
            fig.tight_layout(pad=0.25)
            plt.savefig(f'./results/image_{checkpoint[24:-4]}.jpg')
            plt.close()