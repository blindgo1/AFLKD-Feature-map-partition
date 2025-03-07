import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from Cloud_model import Vgg16Net
from Edge_model import Vgg_small
import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import warnings
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
training_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# training_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
# test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# 限制训练集数据量到前 2000 张
subset_indices = list(range(500))  # 创建索引列表
limited_training_set = Subset(training_set, subset_indices)
subset_indices = list(range(1000))  # 创建索引列表
limited_test_set = Subset(test_set, subset_indices)

# 下载并加载MNIST数据集
# full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# #选择训练数据数量
# num_samples = 1000
# train_dataset, _ = random_split(full_train_dataset, [num_samples, len(full_train_dataset) - num_samples])

#test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 定义超参数和训练参数
batch_size = 32  # 批处理大小
num_epochs = 20  # 训练轮数（epoch）
learning_rate = 0.01  # 学习率（learning rate）

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # 判断是否使用GPU进行训练，如果有GPU则使用第一个GPU（cuda:0）进行训练，否则使用CPU进行训练。

# 定义数据加载器
train_loader = DataLoader(limited_training_set, batch_size=32, shuffle=True)
test_loader = DataLoader(limited_test_set, batch_size=32, shuffle=False)
state_dict = torch.load('pth/model_cloud_CIFAR10.pth')
#加载教师模型
teacher_model = Vgg16Net().to(device)
teacher_model.load_state_dict(state_dict, strict=True)
teacher_model.eval()
#加载学生模型共享前几层参数
def load_pretrained_weights(student_model, teacher_model):

    # Map weights for the first two conv blocks
    student_model.layer1[0].weight.data = teacher_model.layer1[0].weight.data
    student_model.layer1[0].bias.data = teacher_model.layer1[0].bias.data

    student_model.layer1[3].weight.data = teacher_model.layer1[3].weight.data
    student_model.layer1[3].bias.data = teacher_model.layer1[3].bias.data

    student_model.layer1[6].weight.data = teacher_model.layer1[6].weight.data
    student_model.layer1[6].bias.data = teacher_model.layer1[6].bias.data

    # student_model.layer2[0].weight.data = teacher_model.layer2[0].weight.data
    # student_model.layer2[0].bias.data = teacher_model.layer2[0].bias.data
    #
    # student_model.layer2[3].weight.data = teacher_model.layer2[3].weight.data
    # student_model.layer2[3].bias.data = teacher_model.layer2[3].bias.data

student_model = Vgg_small().to(device)
#加载冻结参数
load_pretrained_weights(student_model, teacher_model)
# # #冻结网络参数
# for param in student_model.layer1.parameters():
#     param.requires_grad = False
# for param in student_model.layer2.parameters():
#     param.requires_grad = False

#定义蒸馏参数
temp = 10  # 蒸馏温度
hard_loss = nn.CrossEntropyLoss()
alpha = 1  #hard_loss权重
soft_loss = nn.KLDivLoss(reduction='batchmean')
# 训练模型
optimizer = optim.SGD(student_model.parameters(), lr=learning_rate, momentum=0.9)  # 使用随机梯度下降优化器（SGD）
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # 将图像数据移动到指定设备
        labels = labels.to(device)  # 将标签数据移动到指定设备
        with torch.no_grad():
            teachers_preds = teacher_model(images) #教师网络预测结果
        # 学生模型预测
        students_preds = student_model(images)
        # 计算hard_loss
        students_loss = hard_loss(students_preds, labels)
        # 计算蒸馏后的预测结果及soft_loss
        distillation_loss = soft_loss(
            nn.functional.log_softmax(students_preds / temp, dim=1),
            nn.functional.softmax(teachers_preds / temp, dim=1)
        )
        # 将hard_loss和soft_loss加权求和
        loss = alpha * students_loss + (1 - alpha) * distillation_loss * temp * temp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 训练结束，保存模型参数
torch.save(student_model.state_dict(), 'pth/model_Edge_CIFAR10.pth')

# 加载训练好的模型参数
student_model.load_state_dict(torch.load('pth/model_Edge_CIFAR10.pth'))
student_model.eval()  # 将模型设置为评估模式，关闭dropout等操作

# 定义评估指标变量
correct = 0  # 记录预测正确的样本数量
total = 0  # 记录总样本数量

# 测试模型性能
with torch.no_grad():  # 关闭梯度计算，节省内存空间
    for images, labels in test_loader:
        images = images.to(device)  # 将图像数据移动到指定设备
        labels = labels.to(device)  # 将标签数据移动到指定设备
        outputs = student_model(images)  # 模型前向传播，得到预测结果
        _, predicted = torch.max(outputs.data, 1)  # 取预测结果的最大值对应的类别作为预测类别
        total += labels.size(0)  # 更新总样本数量
        correct += (predicted == labels).sum().item()  # 统计预测正确的样本数量

# 计算模型准确率并打印出来
accuracy = 100 * correct / total  # 计算准确率，将正确预测的样本数量除以总样本数量并乘以100得到百分比形式的准确率。
print('Accuracy of the model on the test images: {} %'.format(accuracy))  # 打印出模型的准确率。