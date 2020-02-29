import torch
import numpy as np
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
        return self.linear(x)
    
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
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(p=0.5, inplace=False),
        )
        
        self.output_layer = nn.Linear(128, num_classes)
    
    def get_feature_vetor(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 512)
        return self.linear(x)

    def forward(self, x):
        return self.output_layer(self.get_feature_vetor(x))
        
    
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
        return self.linear(x)
    
class Custom_10(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(Custom_10, self).__init__()
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
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.downsample_16to8 = nn.Conv2d(32, 64, kernel_size=1, stride=2)
        # 8 * 8
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.downsample_8to4 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        # 4 * 4
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        self.downsample_4to2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
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
        x = F.leaky_relu(self.conv_2(x) + self.downsample_16to8(x))
        x = F.leaky_relu(self.conv_3(x) + self.downsample_8to4(x))
        x = F.leaky_relu(self.conv_4(x) + self.downsample_4to2(x))
        x = x.view(-1, 1024)
        return self.linear(x)

class hierarchical_model(nn.Module):
    def __init__(self, classes, models):
        super(hierarchical_model, self).__init__()
        print(classes)
        self.classes = classes
        self.num_class = len(classes[0])
        self.models = models
        parameters = []
        for m in models:
            parameters+=list(m.parameters())
        self.myparameters = nn.ParameterList(parameters)
        
    def forward(self, x):
        minor_porb= 0.01
        batch_size = x.shape[0]
        result = torch.zeros((batch_size, self.num_class))
        result[:,:] = np.log(minor_porb)
        determined = torch.zeros(batch_size)
        
        x = x.float().to(torch.device('cpu'))
        
        for i in range(0,len(self.models)):
            c = self.classes[i]
            self.models[i].eval()
            temp = self.models[i](x).data
            max_index = torch.argmax(temp, axis=1)
            for j in range(temp.shape[1]):
                if (np.sum(c==j) == 1):
                    new_determined = (max_index==j) & (determined==0)
                    result[new_determined, c==j] = np.log(1-minor_porb*(self.num_class-1))
                    determined[new_determined] = 1
        return result
    
    # def forward(self, x):
    #     batch_size = x.shape[0]
    #     result = torch.zeros((batch_size, self.num_class))
    #     
    #     for i in range(0,len(self.models)):
    #         c = self.classes[i]
    #         self.models[i].eval()
    #         temp = self.models[i](x).data
    #         for j in range(temp.shape[1]):
    #             result[:, c==j] += temp[:,j].view(batch_size, 1)
    #     return result

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
    elif model_name == 'Custom_10':
        custom_conv_net = Custom_10(num_of_channels, num_classes)
    else:
        raise ModelNotFound
    return custom_conv_net
