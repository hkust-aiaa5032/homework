import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
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

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
])

test_dataset = MyDataset("video_frames_30fpv_320", "test_for_student.csv", transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Model
net = Net().to(device)
net.load_state_dict(torch.load('model_best.pth'))

# Evaluation
net.eval()
result = []
with torch.no_grad():

        
fread = open("test_for_student.label", "r")
video_ids = []
for line in fread.readlines():
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)


with open('result.csv', "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(result):
        f.writelines("%s,%d\n" % (video_ids[i], pred_class))


