import os
import copy
import getpass
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

CIFAR_DIR = f'/data5/{getpass.getuser()}/cifar10'
if not os.path.isdir(CIFAR_DIR):
    os.makedirs(CIFAR_DIR)


class CIFAR10(data.Dataset):

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        self.dataset = datasets.cifar.CIFAR10(
            CIFAR_DIR, 
            train=True,
            download=True,
            transform=image_transforms,
        )

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        return (index, img_data.float(), label)

    def __len__(self):
        return len(self.dataset)


class CIFAR10TwoViews(CIFAR10):

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        return index, img_data.float(), img2_data.float(), label
