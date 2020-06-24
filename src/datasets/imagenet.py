"""
Loader for ImageNet. Borrowed heavily from
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import os
import copy
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

IMAGENET_DIR = '/data5/chengxuz/Dataset/imagenet_raw'


class ImageNet(data.Dataset):

    def __init__(
            self, 
            train=True, 
            imagenet_dir=IMAGENET_DIR, 
            image_transforms=None,
        ):
        super().__init__()
        split_dir = 'train' if train else 'validation'
        self.imagenet_dir = os.path.join(imagenet_dir, split_dir)
        self.dataset = datasets.ImageFolder(self.imagenet_dir, image_transforms)

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        return (index, img_data.float(), label)

    def __len__(self):
        return len(self.dataset)
