# coding=utf-8
"""Models."""

import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,dropout_prob=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 256 * 56 * 56)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_prob)
        x = self.fc2(x)
        return x

