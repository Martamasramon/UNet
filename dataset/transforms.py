import numpy as np
import torch
from torchvision import transforms as T
from skimage import transform
from PIL import Image
import random

class Resize(object):
  def __init__(self, output_size):
    self.output_size  = output_size

  def __call__(self, image):
    return transform.resize(image, (self.output_size, self.output_size),preserve_range=True)


class CenterCrop(object):
  def __init__(self, output_size):
    self.output_size = (output_size, output_size)

  def __call__(self, image):
    h, w = image.shape[:2]
    new_h, new_w = self.output_size

    top  = (h - new_h) // 2
    left = (w - new_w) // 2
    
    return image[top: top + new_h, left: left + new_w]


class ToTensor(object):
  def __call__(self, image):
    if len(image.shape)==2:
      image = np.expand_dims(image, axis=0)
    
    return torch.from_numpy(image).float()
  
  
class RandomHorFlip(object):
  def __init__(self, p=0.5):
    self.p = p
    
  def __call__(self, image):
    if random.random() > self.p:
      image = np.fliplr(image).copy()
    return image


def get_train_transform(img_size=64):
  if img_size==64:
    return T.Compose([
        CenterCrop(128),
        Resize(64),
        RandomHorFlip(),
        ToTensor(),
    ])
  else:
    return T.Compose([
        CenterCrop(img_size),
        RandomHorFlip(),
        ToTensor(),
    ])

def get_test_transform(img_size=64):
  if img_size==64:
    return T.Compose([
        CenterCrop(128),
        Resize(64),
        ToTensor(),
    ])
  else:
    return T.Compose([
        CenterCrop(img_size),
        ToTensor(),
    ])