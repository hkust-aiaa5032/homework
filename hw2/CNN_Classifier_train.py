import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
from models import Net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
class MyDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.classes = sorted(self.df[1].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        vid, label = self.df.iloc[index, :]
        img_list = os.listdir(os.path.join(self.root, f"{vid}"))
        img_list = sorted(img_list)
        img_path = os.path.join(self.root, f"{vid}", img_list[int(len(img_list)/2)])
 
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes.index(label)
        return img, label

# You can add data augmentation here
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


trainval_dataset = MyDataset("/video_frames_30fpv_320p", "/trainval.csv", transform)
train_data, val_data = train_test_split(trainval_dataset, test_size=0.2, random_state=0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

net = Net().to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()


for epoch in range(50):
    # Metrics here ...

    
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # Training code ...

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            # Validation code ...


    torch.save(net.state_dict(), 'model_last.pth')
