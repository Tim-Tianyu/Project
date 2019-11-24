import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Custume_07(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc1_bn= nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.fc2_bn= nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv4(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)
    
class Custume_04(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

Resnet_18 = models.resnet18

VGG_11 = models.vgg11_bn