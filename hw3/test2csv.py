import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from dataset import MyDataset
from models import resnet18


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])


test_dataset = MyDataset("/hw3_16fpv", "test_for_student.csv", stage="test",ratio=0.2,transform=transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(len(test_loader))


net = resnet18(num_classes=10, sample_size=224, sample_duration=16).to(device)
net.load_state_dict(torch.load('ResNet18_best.pth'))



net.eval()
result = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        result.extend(predicted.cpu().numpy())
        
fread = open("test_for_student.label", "r")
video_ids = []
for line in fread.readlines():
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)



with open('result_ResNet18_3D.csv', "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(result):
        f.writelines("%s,%d\n" % (video_ids[i], pred_class))


