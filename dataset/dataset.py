from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

import matplotlib.pyplot as plt

class MyDataset(Dataset):
  def __init__(self, img_path, transforms, use_histo=False, use_t2w=False, is_pretrain=True, is_train=True):
    root   = 'pretrain' if is_pretrain else 'finetune'
    suffix = 'train'    if is_train    else 'test'
    self.img_path   = img_path
    self.img_dict   = pd.read_csv(f'../Dataset_preparation/{root}_{suffix}.csv')
    self.transforms = transforms
    self.use_histo  = use_histo
    self.use_t2w    = use_t2w

  def __len__(self):
    return len(self.img_dict)

  def __getitem__(self, idx):
    item   = self.img_dict.iloc[idx]
    img    = Image.open(self.img_path + item["SID"])
    img    = np.array(img)/255.0
    sample = {'image': img, 'label': img}
    
    sample = self.transforms(sample)
            
    return sample

  def __len__(self):
    return len(self.img_dict)
