import torch.nn as nn


class Vgg16Net(nn.Module):
    def __init__(self):
        super(Vgg16Net, self).__init__()

        self.layer1 = nn.Sequential(
            # （输入通道，输出通道，卷积核大小） 例：32*32*3 —> (32+2*1-3)/1+1 = 32，输出：32*32*32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # （输入通道，输出通道，卷积核大小） 输入：32*32*48，卷积：3*48*48，输出：32*32*48
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 输入：32*32*32，输出：16*16*32
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)#输出：8*8*48
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 80, 3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),

            nn.Conv2d(80, 80, 3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),

            nn.Conv2d(80, 80, 3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),

            nn.Conv2d(80, 80, 3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),

            nn.Conv2d(80, 80, 3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2) #输出：4*4*80
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(80, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2) #输出：2*2*128
        )


        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

        self.fc = nn.Sequential(
            nn.Linear(128*2*2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 100),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x