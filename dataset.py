import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision      import transforms as T
import pandas as pd

class MyDataset(Dataset):
  def __init__(self, 
        img_path, 
        img_size    = 64, 
        use_histo   = False, 
        use_t2w     = False, 
        use_mask    = False,
        is_finetune = False, 
        is_train    = True):
    
    root   = 'finetune' if is_finetune else 'pretrain'
    suffix = 'train'    if is_train    else 'test'
    
    self.masked     = '_mask' if use_mask else ''
    self.img_path   = img_path
    self.img_dict   = pd.read_csv(f'../Dataset_preparation/{root}{self.masked}_{suffix}.csv')
    self.use_histo  = use_histo
    self.use_t2w    = use_t2w
    
    if is_train:
      self.transform = T.Compose([
        T.CenterCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor()
      ]) 
    else:
      self.transform = T.Compose([
        T.CenterCrop(img_size),
        T.ToTensor()
      ]) 

  def __len__(self):
    return len(self.img_dict)

  def __getitem__(self, idx):
    item   = self.img_dict.iloc[idx]
    img    = Image.open(f'{self.img_path}/T2W{self.masked}/{item["SID"]}').convert('L')                
    return self.transform(img)

  def __len__(self):
    return len(self.img_dict)
