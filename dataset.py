from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import glob
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.cluster import KMeans
import json
import torch
import random


class SatelliteToMapDataset(Dataset):
    def __init__(self, root_dir, resize=None, augmentation=True, plot=False, format='CMYK', n_samples_per_image=5000, load=True):
        self.root_dir = root_dir
        self.load = load
        self.format = format
        self.n_samples_per_image = n_samples_per_image
        self.file_list = glob.glob(self.root_dir + '/*.jpg')
        self.transform = self.get_transforms(resize, augmentation)
        self.plot = plot
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        w, h = img.size
        
        img = img.convert(self.format)
        
        img_A, img_B = self.transform_set(img.crop((0, 0, w / 2, h)), img.crop((w / 2, 0, w, h)), random.randint(0, 2**32))

        return {'input': img_A, 'output': img_B}
    
    def transform_set(self, img_A, img_B, seed):

        random.seed(seed)
        torch.manual_seed(seed)

        img_A = self.transform(img_A)

        random.seed(seed)
        torch.manual_seed(seed)

        img_B = self.transform(img_B)

        return img_A, img_B
    
    def get_transforms(self, resize=None, augmentation=True):
        transform_list = []

        if resize is not None:
            transform_list.append(transforms.Resize(resize))

        transform_list.append(transforms.ToTensor())

        if augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.4),
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])
            
        num_channels = 4 if self.format == 'CMYK' else 3
        transform_list.append(transforms.Normalize([0.5] * num_channels, [0.5] * num_channels))

        return transforms.Compose(transform_list)