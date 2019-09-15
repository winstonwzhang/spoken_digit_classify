import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
  '''Main neural net model.'''
  def __init__(self):
    super(BaseModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3)
    self.bn1 = nn.BatchNorm2d(24)
    self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
    
    self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
    self.conv2o = nn.Conv2d(in_channels=48, out_channels=24, kernel_size=1)
    self.bn2 = nn.BatchNorm2d(24)
    self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
    
    self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
    self.conv3o = nn.Conv2d(in_channels=48, out_channels=24, kernel_size=1)
    self.bn3 = nn.BatchNorm2d(24)
    self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
    
    self.drop1 = nn.Dropout(p=0.2)
    self.fc1 = nn.Linear(in_features=4968, out_features=1024)
    self.drop2 = nn.Dropout(p=0.2)
    self.fc2 = nn.Linear(in_features=1024, out_features=64)
    self.fc3 = nn.Linear(in_features=64, out_features=10)

  def forward(self, x):
    
    x = F.relu(self.conv1(x))
    x = self.pool1(self.bn1(x))
    
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv2o(x))
    x = self.pool2(self.bn2(x))
    
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv3o(x))
    x = self.pool3(self.bn3(x))
    
    x = torch.flatten(x, start_dim=1)
    
    x = self.drop1(x)
    x = F.relu(self.fc1(x))
    
    x = self.drop2(x)
    x = F.relu(self.fc2(x))
    
    x = self.fc3(x)
    return x


def model_predict(mfcc, mdl):
    '''mfcc - numpy 2d array of mfcc, mdl - model'''
    with torch.no_grad():
        img = torch.from_numpy(mfcc)
        img = img.unsqueeze_(0).unsqueeze_(0)  # add singleton dimensions
        
        # feed mfcc to model
        output = mdl(img)
        # model output hasn't been softmaxed
        output = F.softmax(output, dim=1)
        class_preds = output.detach().numpy()

    return class_preds
