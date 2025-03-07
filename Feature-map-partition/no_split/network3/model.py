import time

import torch
import torch.nn as nn

#part1
class VGG16Small_part1(nn.Module):
    def __init__(self):
        super(VGG16Small_part1, self).__init__()
        # Conv Block 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        for name, layer in self.named_children():
            start_time = time.time()
            x = layer(x)
            end_time = time.time()
            print(name, end_time - start_time)
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.maxpool1(x)
        return x
#part2
class VGG16Small_part2(nn.Module):
    def __init__(self):
        super(VGG16Small_part2, self).__init__()
        # Conv Block 2
        self.conv4 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        for name, layer in self.named_children():
            start_time = time.time()
            x = layer(x)
            end_time = time.time()
            print(name, end_time - start_time)
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.maxpool2(x)
        return x


class VGG16Small_part3(nn.Module):
    def __init__(self):
        super(VGG16Small_part3, self).__init__()
        # Conv Block 2
        self.conv6 = nn.Conv2d(48, 80, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(80, 80, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        for name, layer in self.named_children():
            start_time = time.time()
            x = layer(x)
            end_time = time.time()
            print(name, end_time - start_time)
        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.maxpool2(x)
        return x

#part3
class VGG16Small_part4(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16Small_part4, self).__init__()
        self.classifier = nn.Sequential(
        nn.Linear(80 * 28 * 28, 1024),  # Adjust input size for 32x32 input
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        start_time = time.time()
        x = x.view(-1)
        # print('x:', x.shape)
        x = self.classifier(x)
        end_time = time.time()
        print('fc_time:', end_time - start_time)
        return x

def create_model_part2_instance():
    return VGG16Small_part2().to('cpu')
def create_model_part3_instance():
    return VGG16Small_part3().to('cpu')
def create_model_part4_instance():
    return VGG16Small_part4().to('cpu')