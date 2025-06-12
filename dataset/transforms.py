import numpy as np
import torch
from torchvision import transforms
from skimage import transform
from PIL import Image
import random

class CenterCrop(object):
  def __init__(self, output_size):
    self.output_size = (output_size, output_size)

  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    h, w = image.shape[:2]
    new_h, new_w = self.output_size

    top  = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[top: top + new_h, left: left + new_w]
    label = label[top: top + new_h, left: left + new_w]

    sample['image'] = image
    sample['label'] = label
    return sample


class ToTensor(object):
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    if len(image.shape)==2:
      image = np.expand_dims(image, axis=0)
      label = np.expand_dims(label, axis=0)
    
    sample['image'] = torch.from_numpy(image).float()
    sample['label'] = torch.from_numpy(label).float()
    return sample


  
class RandomHorFlip(object):
  def __init__(self, p=0.5):
    self.p = p
    
  def __call__(self, sample):
    if random.random() > self.p:
      sample['image'] = np.fliplr(sample['image']).copy()
      sample['label'] = np.fliplr(sample['label']).copy()
    return sample

def get_train_transform(img_size=64, scale_factor=2, same_size_input_label=True):
  transform = transforms.Compose([
      CenterCrop(img_size),
      RandomHorFlip(),
      ToTensor(),
  ])
  return transform

def get_test_transform(img_size=64, scale_factor=2, same_size_input_label=True):
  transform = transforms.Compose([
      CenterCrop(img_size),
      ToTensor(),
  ])
  return transform

  