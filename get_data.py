import os
from skimage import io
import pandas as pd
import numpy as np
import torch
import random
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch.utils.data as Data
from PIL import Image

class ISIC2019(Dataset):
    def __init__(self, csv_file, root_dir, image_size, mode):
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_arr = np.asarray(self.file.iloc[:, 1])
        self.imagesz = image_size
        self.mode = mode

    def __len__(self):
        return len(self.file)

    def transform(self, image_size):
        trans = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return trans

    def transform_test(self, image_size):
        trans = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return trans

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.label_arr[idx]
        sample = image
        trans = self.transform(label, self.imagesz)
        trans_test = self.transform_test(label, self.imagesz)
        if self.transform:
            if mode == 'test':
                sample = trans_test(sample)
            else:
                sample = trans(sample)
        return sample, label

def train_loader(imagesz, batchsz):
    mode = 'train'
    csv_file = './data/txt_file/train.txt'
    root_dir = './data/ISIC2019_train'
    data = ISIC2019(csv_file, root_dir, imagesz, mode)

    loader = Data.DataLoader(data, batch_size=batchsz, shuffle=True, num_workers=8)
    return loader
    
def val_loader(imagesz, batchsz):

    mode = 'val'
    csv_file = './data/txt_file/val.txt'
    root_dir = './data/ISIC2019_train'
    data = ISIC2019(csv_file, root_dir, imagesz, mode)

    loader = Data.DataLoader(data, batch_size=batchsz, shuffle=True, num_workers=8)
    return loader


def test_loader(imagesz, batchsz):
    mode = 'test'
    csv_file = './data/txt_file/val.txt'
    root_dir = './data/ISIC2019_train'
    data = ISIC2019(csv_file, root_dir, imagesz, mode)

    loader = Data.DataLoader(data, batch_size=batchsz, shuffle=True, num_workers=8)
    return loader