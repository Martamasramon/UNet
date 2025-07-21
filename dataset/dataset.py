from PIL import Image
from torch.utils.data import Dataset
from torchvision      import transforms as T
import pandas as pd
from dataset.transforms  import get_train_transform, get_test_transform
import numpy as np

class MyDataset(Dataset):
  def __init__(self, 
        img_path, 
        data_type,
        img_size    = 64, 
        use_mask    = False,
        is_finetune = False, 
        is_train    = True):
    
    root   = 'finetune' if is_finetune else 'pretrain'
    
    self.masked     = '_mask' if use_mask else ''
    self.img_path   = img_path
    self.img_dict   = pd.read_csv(f'../Dataset_preparation/{root}{self.masked}_{data_type}.csv')
    self.transform  = get_train_transform(img_size) if data_type=='train' else get_test_transform(img_size)

    print(f'Loading data from {root}{self.masked}_{data_type}...')
    

  def __len__(self):
    return len(self.img_dict)

  def __getitem__(self, idx):
    item   = self.img_dict.iloc[idx]
    img    = Image.open(f'{self.img_path}/T2W{self.masked}/{item["SID"]}').convert('L')                
    img    = np.array(img)/255.0
    
    return self.transform(img)

  def __len__(self):
    return len(self.img_dict)
