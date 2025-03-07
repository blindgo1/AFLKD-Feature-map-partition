import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.nn.functional as F
from store_node import save_padding,retrieve_padding
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#part1
class VGG16Small_part1(nn.Module):
    def __init__(self):
        super(VGG16Small_part1, self).__init__()
        # Conv Block 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x,rank):
        #i为卷积层的层数
        i = 0
        for name, layer in self.named_children():
            start_time = time.time()
            #判断如果是卷积层
            if isinstance(layer, nn.Conv2d):
                if rank == 0:
                    # 判断上一层的padding是否存储完毕
                    while rpc.rpc_sync(f"raspberrypi8", retrieve_padding, args=(f'conv{i}_{rank + 1}',)) is None:
                        pass
                    # future_padding = rpc.rpc_async(f"raspberrypi8", retrieve_padding, args=(f'conv{i}_{rank + 1}',))
                    # padding = future_padding.wait()
                    #获取padding数据并拼接
                    padding = rpc.rpc_sync(f"raspberrypi8", retrieve_padding, args=(f'conv{i}_{rank + 1}',))

                    x = torch.cat((x, padding.to(device)), dim=2)
                    #加填充保证卷积后尺寸与原来一致
                    x = F.pad(x, (1, 1, 1, 0), mode='constant', value=0)
                    #进行卷积计算

                    x = layer(x)

                    #将计算结果的padding送到存储节点中
                    rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv{i+1}_{rank}',
                                                                       x[:, :, -1, :].unsqueeze(2).to('cpu')))
                    i = i + 1

                if rank == 1:
                    # 判断上一层的padding是否存储完毕
                    while rpc.rpc_sync(f"raspberrypi8", retrieve_padding, args=(f'conv{i}_{rank - 1}',)) is None:
                        pass
                    # 获取padding数据并拼接
                    padding = rpc.rpc_sync(f"raspberrypi8", retrieve_padding,
                                           args=(f'conv{i}_{rank - 1}',))
                    x = torch.cat((padding.to(device), x), dim=2)
                    # 加填充保证卷积后尺寸与原来一致
                    x = F.pad(x, (1, 1, 0, 1), mode='constant', value=0)
                    # 进行卷积计算

                    x = layer(x)

                    # 将计算结果的padding送到存储节点中
                    rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv{i + 1}_{rank}',
                                                                       x[:, :, 0, :].unsqueeze(2).to('cpu')))
                    i = i + 1
            else:
                x = layer(x)
            end_time = time.time()
            print(name, end_time - start_time)
        return x
#part2
class VGG16Small_part2(nn.Module):
    def __init__(self):
        super(VGG16Small_part2, self).__init__()
        # Conv Block 2
        self.conv4 = nn.Conv2d(32, 48, kernel_size=3, padding=0)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, padding=0)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, rank):
        # i为卷积层的层数，第二部分卷积层从conv4开始
        i = 3
        for name, layer in self.named_children():
            start_time = time.time()
            # 判断如果是卷积层
            if isinstance(layer, nn.Conv2d):
                if rank == 2:
                    while rpc.rpc_sync(f"raspberrypi8", retrieve_padding, args=(f'conv{i}_{rank + 1}',)) is None:
                        pass
                    padding = rpc.rpc_sync(f"raspberrypi8", retrieve_padding,
                                           args=(f'conv{i}_{rank + 1}',))

                    x = torch.cat((x, padding), dim=2)
                    # 加填充保证卷积后尺寸与原来一致
                    x = F.pad(x, (1, 1, 1, 0), mode='constant', value=0)
                    # 进行卷积计算
                    x = layer(x)
                    end_time = time.time()
                    # 将计算结果的padding送到存储节点中
                    rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv{i + 1}_{rank}',
                                                                       x[:, :, -1, :].unsqueeze(2).to('cpu')))
                    i = i + 1

                if rank == 3:
                    while rpc.rpc_sync(f"raspberrypi8", retrieve_padding, args=(f'conv{i}_{rank - 1}',)) is None:
                        pass
                    padding = rpc.rpc_sync(f"raspberrypi8", retrieve_padding,
                                           args=(f'conv{i}_{rank - 1}',))

                    x = torch.cat((padding,x), dim=2)
                    # 加填充保证卷积后尺寸与原来一致
                    x = F.pad(x, (1, 1, 0, 1), mode='constant', value=0)
                    # 进行卷积计算
                    x = layer(x)
                    end_time = time.time()
                    # 将计算结果的padding送到存储节点中
                    rpc.rpc_sync(f"raspberrypi8", save_padding, args=(f'conv{i + 1}_{rank}',
                                                                       x[:, :, -1, :].unsqueeze(2).to('cpu')))
                    i = i + 1
            else:
                x = layer(x)
                end_time = time.time()
            print(name, end_time - start_time)
        return x


#part3
class VGG16Small_part3(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16Small_part3, self).__init__()
        self.classifier = nn.Sequential(
        nn.Linear(48 * 56 * 56, 1024),  # Adjust input size for 32x32 input
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        start_time = time.time()
        x = x.view(-1)
        #print('x:', x.shape)
        x = self.classifier(x)
        end_time = time.time()
        print('fc_time:', end_time - start_time)
        return x

def create_model_part2_instance():
    return VGG16Small_part2().to('cpu')
def create_model_part3_instance():
    return VGG16Small_part3().to('cpu')