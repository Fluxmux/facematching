import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from resnet import *


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),
            
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),


            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),
        )
        

        self.fc1 = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)   
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    
class SiameseNetworkConcat(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1, dilation=1, return_indices=False, ceil_mode=False),
            
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1, dilation=1, return_indices=False, ceil_mode=False),


            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),
            
        )
        

        self.fc1 = nn.Sequential(
            nn.Linear(1600, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 10),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(inplace=True),

            nn.Linear(20, 1),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)   
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = F.relu(self.forward_once(input1))
        output2 = F.relu(self.forward_once(input2))
        output = torch.cat((output1, output2), dim=1)
        output = self.fc2(output)
        return output
    
    
class SiameseNetworkAbs(nn.Module):
    def __init__(self):
        super(SiameseNetworkAbs, self).__init__()
    
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 16, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1, dilation=1, return_indices=False, ceil_mode=False),
            
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1, dilation=1, return_indices=False, ceil_mode=False),


            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False),
            
        )
        

        self.fc1 = nn.Sequential(
            nn.Linear(1600, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 10),
        )
        
        self.fc2 = nn.Linear(10, 1)
        

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        dist = torch.abs(output1 - output2)
        output = self.fc2(dist)
        return output

    
class SiameseResNet(ResNet):
    def __init__(self):
        super(SiameseResNet, self).__init__(BasicBlock, [3], num_classes = 16)
        
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(7, 7), stride=(3, 3), padding=(0, 0), bias=False)
       
    def forward_once(self, x):  
        return super(SiameseResNet, self).forward(x)
            
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    
class SiameseResNetConcat(ResNet):
    def __init__(self):
        super(SiameseResNetConcat, self).__init__(BasicBlock, [3], num_classes = 16)
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(7,7 ), stride=(3, 3), padding=(0, 0), bias=False)
        
        self.fc1 = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64,256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64,1),
            
        )
        
    def forward_once(self, x):
        return super(SiameseResNetConcat, self).forward(x)
            
    def forward(self, input1, input2):
        output1 = F.relu(self.forward_once(input1))
        output2 = F.relu(self.forward_once(input2))
        output = torch.cat((output1, output2), dim=1)
        output = self.fc1(output)
        return output
        