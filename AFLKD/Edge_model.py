import torch.nn as nn


class Vgg_small(nn.Module):
    def __init__(self):
        super(Vgg_small, self).__init__()

        self.layer1 = nn.Sequential(
            # （输入通道，输出通道，卷积核大小） 例：32*32*3 —> (32+2*1-3)/1+1 = 32，输出：32*32*32
            nn.Conv2d(1, 32, 3, padding=1),
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

        # self.conv_layer = nn.Sequential(
        #     self.layer1,
        #     self.layer2,
        # )

        self.fc_edge = nn.Sequential(
            nn.Linear(80*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_edge(x)
        return x