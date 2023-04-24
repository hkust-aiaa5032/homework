import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import random

class MyDataset(Dataset):
    def __init__(self, root, csv_file,stage="train",ratio=0.2, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.stage = stage
        self.ratio = ratio
        self.files = None
        filenames = self.df[0].unique()
        if self.stage == "test":
            self.files = filenames
        else:
            filenames = sorted(filenames)
            length = len(filenames)
            train_files = filenames[int(length * self.ratio):]
            val_files = filenames[:int(length * self.ratio)]
            if self.stage == "train":
                self.files = train_files
            elif self.stage == "val":
                self.files = val_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        vid = self.files[index]
        vid = vid.split('.mp4')[0]
        label = self.df.loc[self.df[0] == vid, 1].values[0]
        img_list = os.listdir(os.path.join(self.root, f"{vid}.mp4"))
        img_list = sorted(img_list)
        img_16fpv = []    
        for i in range(len(img_list)):
            img_path = os.path.join(self.root, f"{vid}.mp4", img_list[i])
            img = Image.open(img_path).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            img_16fpv.append(img)
        img_16fpv_tensor = torch.stack(img_16fpv).permute(1,0,2,3)
        return img_16fpv_tensor, label

