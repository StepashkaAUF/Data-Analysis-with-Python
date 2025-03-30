import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().init()

        self.conv1 = nn.Conv2d(3, 64, (3,3))
        self.conv2 = nn.Conv2d(64, 128, (3,3))
        self.pool1 = nn.MaxPool2d((2,2))
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(128, 256, (3,3))
        self.conv4 = nn.Conv2d(256, 128, (3,3))
        self.conv5 = nn.Conv2d(128, 64, (3,3))
        self.conv6 = nn.Conv2d(64, 32, (3,3))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32, 100)
        self.fc2 = nn.Linear(100, 10)

    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

def create_model():
    return ConvNet()
