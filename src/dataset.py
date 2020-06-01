import os

import numpy as np
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset, DataLoader

from .utils import picture_resize


def augmentation(mode='train'):
    if mode == 'train':
        transform = [
                     albu.OneOf([
                                 albu.GaussianBlur(blur_limit=3, p=1.0),
                                 albu.MotionBlur(blur_limit=3, p=1.0),
                     ], p=0.2),
                     ToTensor(),
        ]

    elif mode == 'valid' or mode == 'test':
        transform = [
                     ToTensor(),
        ]

    else:
        raise TypeError('The argument of augmentaion is unexpected.')

    return albu.Compose(transform)


class TrainDataset(Dataset):
    def __init__(self, img_ids, img_dir, mode='train', size=320, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mode = mode
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx][:-4]

        flip = np.random.randint(2)

        img_path = os.path.join(self.img_dir, img_id+'.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = picture_resize(img, size=self.size)
        if self.mode=='train' and flip:
            img = img[:, ::-1, :]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, img_id


class TestDataset(Dataset):
    def __init__(self, img_ids, img_dir, size=320, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx][:-4]

        img_path = os.path.join(self.img_dir, img_id+'.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = picture_resize(img, size=self.size)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        img = img[:1, :, :]
        img = img * 2 - 1
        
        return img, img_id