import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import warnings
from torchvision.datasets import CIFAR10
from Cloud_model import Vgg16Net

# 定义超参数和训练参数
batch_size = 32  # 批处理大小
num_epochs = 100  # 训练轮数（epoch）
learning_rate = 0.01  # 学习率（learning rate）
best_acc = 0
warnings.filterwarnings("ignore")

# 定义数据预处理操作
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 下载并加载CIFAR10数据集
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#num_classes = 10  # 类别数（MNIST数据集有10个类别）
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU进行训练，如果有GPU则使用第一个GPU（cuda:0）进行训练，否则使用CPU进行训练。

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True
                                          )  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
model = Vgg16Net().to(device)  # 将模型移动到指定设备（GPU或CPU）
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)  # 使用随机梯度下降优化器（SGD）

# 训练模型
model.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # 将图像数据移动到指定设备
        labels = labels.to(device)  # 将标签数据移动到指定设备

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度缓存
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新权重参数

        if (i + 1) % 100 == 0:  # 每100个batch打印一次训练信息
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader),
                                                                     loss.item()))



    # 定义评估指标变量
    correct = 0  # 记录预测正确的样本数量
    total = 0  # 记录总样本数量

    # 测试模型性能
    model.eval()
    with torch.no_grad():  # 关闭梯度计算，节省内存空间
        for images, labels in test_loader:
            images = images.to(device)  # 将图像数据移动到指定设备
            labels = labels.to(device)  # 将标签数据移动到指定设备
            outputs = model(images)  # 模型前向传播，得到预测结果
            _, predicted = torch.max(outputs.data, 1)  # 取预测结果的最大值对应的类别作为预测类别
            total += labels.size(0)  # 更新总样本数量
            correct += (predicted == labels).sum().item()  # 统计预测正确的样本数量
        accuracy = 100 * correct / total
        print('Epoch:{}\t Accuracy:{:.4f}'.format(epoch + 1, accuracy))  # 打印出模型的准确率。
        if accuracy > best_acc:
            f3 = open("best_acc_cloud.txt", "w")
            f3.write(f"训练轮次为{epoch + 1}时,准确率最高!准确率为{accuracy}")
            print(f"训练轮次为{epoch + 1}时,准确率最高!准确率为{accuracy}")
            f3.close()
            best_acc = accuracy
            # 训练结束，保存模型参数
            torch.save(model.state_dict(), './pth/model_cloud_CIFAR10.pth')




