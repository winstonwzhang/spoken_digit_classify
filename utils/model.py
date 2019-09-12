import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
  '''Main neural net model.'''
  def __init__(self):
    super(BaseModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=2)
    self.bn1 = nn.BatchNorm2d(24)
    # self.pool1 = nn.MaxPool2d(kernel_size=(4, 2))
    
    self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=2)
    self.bn2 = nn.BatchNorm2d(48)
    # self.pool2 = nn.MaxPool2d(kernel_size=(4, 2))
    
    self.conv3 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=2)
    self.bn3 = nn.BatchNorm2d(96)
    self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
    
    self.drop1 = nn.Dropout(p=0.25)
    self.fc1 = nn.Linear(in_features=384, out_features=64)
    self.drop2 = nn.Dropout(p=0.25)
    self.fc2 = nn.Linear(in_features=64, out_features=10)

  def forward(self, x):
    x = self.bn1(F.relu(self.conv1(x)))
    x = self.bn2(F.relu(self.conv2(x)))
    x = self.pool3(self.bn3(F.relu(self.conv3(x))))
    x = self.drop1(torch.flatten(x, start_dim=1))
    x = self.drop2(F.relu(self.fc1(x)))
    x = self.fc2(x)
    return x