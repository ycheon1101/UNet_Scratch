import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


# convert label id to used id
class ConvertGtToUsed(object):
    def __init__(self):
        self.converted_label = {
        0:255, 
        1:255,
        2:255,
        3:255,
        4:255,
        5:255, 
        6:255,
        7:0,
        8:1,
        9:255,
        10:255,
        11:2,
        12:3,
        13:4,
        14:255,
        15:255,
        16:255,
        17:5,
        18:255,
        19:6,
        20:7,
        21:8,
        22:9,
        23:10,
        24:11,
        25:12,
        26:13,
        27:14,
        28:15,
        29:255,
        30:255,
        31:16,
        32:17,
        33:18
    }      

    def __call__(self, gt_tensor):
        gt_tensor = np.array(gt_tensor)
        gt_tensor = torch.tensor(gt_tensor, dtype=torch.int64)
        converted_gt_tensor = torch.empty_like(gt_tensor)

        for original_label, new_label in self.converted_label.items():
            converted_gt_tensor[gt_tensor == original_label] = new_label

        return converted_gt_tensor

# cityscapes dataset
class CityscapesDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): CSV path.
            transform: transform for source image.
            target_transform: transform for gt
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_file_name = self.data_frame.iloc[idx, 0]
        img_name = self.data_frame.iloc[idx, 1]  # source image
        # print(f'img_name: {img_name}')
        label_name = self.data_frame.iloc[idx, 2]  # gt image
        
        # print(f'color name: {color_name}')

        image = Image.open(img_name)
        label = Image.open(label_name)
        

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return img_file_name, image, label


train_csv_file = './cityscape_train.csv'
val_csv_file = './cityscape_val.csv'

# transform for src img
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(572),
    transforms.Normalize([0.28689554, 0.32513303, 0.28389177], [0.18696375, 0.19017339, 0.18720214])
])

# transform for gt img
target_transform = transforms.Compose([
    # ConvertToTensor(),
    ConvertGtToUsed(),
    # CenterCropTensorGT(388)
    transforms.CenterCrop(388)
])

# train_data
train_dataset = CityscapesDataset(csv_file=train_csv_file, transform=transform, target_transform=target_transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

# val_data
val_dataset = CityscapesDataset(csv_file=val_csv_file, transform=transform, target_transform=target_transform)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=1)

# src, gt = next(iter(train_dataloader))

# print(src.shape)
# print(gt.shape)

