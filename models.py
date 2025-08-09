import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # 卷积层+批归一化
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层和dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)  # 轻度dropout
        self.dropout2 = nn.Dropout(0.5)  # 中度dropout

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 适配3次池化后的维度
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # 展平特征
        x = x.view(-1, 128 * 3 * 3)

        # 全连接层+dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # 防止过拟合
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout2d = nn.Dropout2d()  # 用于卷积层后的4D输入
        self.dropout = nn.Dropout()      # 新增：用于全连接层后的2D输入
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout2d(self.conv2(x)), 2))  # 卷积层后用dropout2d
        x = x.view(-1, 320)  # 展平为2D张量
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 全连接层后用普通dropout
        x = self.fc2(x)
        return x


class SimpleFC(nn.Module):
    """简单全连接网络模型（用于对比）"""
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x