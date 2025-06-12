import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from dataset.transforms  import get_train_transform, get_test_transform

class MyDataset(Dataset):
  def __init__(self, img_path, img_size=128, use_histo=False, use_t2w=False, is_pretrain=True, is_train=True):
    root   = 'pretrain' if is_pretrain else 'finetune'
    suffix = 'train'    if is_train    else 'test'
    self.img_path   = img_path
    self.img_dict   = pd.read_csv(f'../Dataset_preparation/{root}_{suffix}.csv')
    self.transforms = get_train_transform(img_size) if is_train else get_test_transform(img_size)
    self.use_histo  = use_histo
    self.use_t2w    = use_t2w

  def __len__(self):
    return len(self.img_dict)

  def __getitem__(self, idx):
    item   = self.img_dict.iloc[idx]
    img    = Image.open(self.img_path + item["SID"])
    img    = np.array(img)/255.0
    sample = {'image': img, 'label': img}
                
    return self.transforms(sample)

  def __len__(self):
    return len(self.img_dict)
