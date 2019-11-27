import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Custom_07(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(Custom_07, self).__init__()
        # 32 * 32
        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace = True)
        )
        # 16 * 16
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace = True)
        )
        # 8 * 8
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True)
        )
        # 4 * 4
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace = True)
        )
        # 2 * 2
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace = True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace = True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.view(-1, 1024)
        x = self.linear(x)
        return F.log_softmax(x)
    
class Custom_05(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(Custom_05, self).__init__()
        # 32 * 32
        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace = True)
        )
        # 14 * 14
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace = True)
        )
        # 6 * 6
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True)
        )
        # 2 * 2 * 128
        self.linear = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace = True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return F.log_softmax(x)
    
class Custom_08(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(Custom_08, self).__init__()
        # 32 * 32
        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace = True)
        )
        # 16 * 16
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace = True)
        )
        # 8 * 8
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True)
        )
        # 4 * 4
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace = True)
        )
        # 2 * 2
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace = True)
        )
        # 1 * 1
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace = True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace = True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return F.log_softmax(x)

class ModelNotFound(Exception):
    pass

def load_model(model_name, num_of_channels, num_classes=10):
    if model_name == 'VGG11':
        custom_conv_net = models.vgg11_bn()
        custom_conv_net.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias = True)
    elif model_name == 'Custom_05':
        custom_conv_net = Custom_05(num_of_channels, num_classes)
    elif model_name == 'Custom_07':
        custom_conv_net = Custom_07(num_of_channels, num_classes)
    elif model_name == 'Custom_08':
        custom_conv_net = Custom_08(num_of_channels, num_classes)
    else:
        raise ModelNotFound
    return custom_conv_net
