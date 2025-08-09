import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
from models import CNN, SimpleFC, ImprovedCNN  # 导入新模型
from utils import get_device, set_chinese_font
import matplotlib.pyplot as plt

# 数据转换 - 增加数据增强
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def train_model(model_type='CNN', epochs=10, batch_size=64, device=None, save_model=True):
    """训练模型并返回训练耗时"""
    if device is None:
        device = get_device()

    # 加载数据集
    train_dataset = datasets.MNIST(
        root='data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root='data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型 - 增加对ImprovedCNN的支持
    if model_type == 'CNN':
        model = CNN().to(device)
        model_path = 'mnist_cnn_model.pth'
    elif model_type == 'SimpleFC':
        model = SimpleFC().to(device)
        model_path = 'mnist_simplefc_model.pth'
    elif model_type == 'ImprovedCNN':  # 新增模型类型
        model = ImprovedCNN().to(device)
        model_path = 'mnist_improvedcnn_model.pth'
    else:
        raise ValueError("model_type必须是'CNN'、'SimpleFC'或'ImprovedCNN'")  # 更新错误提示

    # 定义损失函数、优化器和学习率调度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # 记录训练开始时间
    start_time = time.time()

    # 训练过程
    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    patience = 5
    no_improve_epochs = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'第{epoch + 1}轮 [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

        epoch_time = time.time() - epoch_start
        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'第{epoch + 1}轮训练耗时: {epoch_time:.2f}秒 | 平均损失: {avg_loss:.6f}')

        # 测试集验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'测试集准确率: {accuracy:.2f}%\n')

        # 学习率调度
        scheduler.step(accuracy)

        # 早停判断
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_weights = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"早停于第{epoch + 1}轮，最佳准确率: {best_accuracy:.2f}%")
                break

    # 计算总训练耗时
    total_time = time.time() - start_time
    print(f'===== 训练完成 =====')
    print(f'总训练轮次: {epoch + 1}')
    print(f'总耗时: {total_time:.2f}秒')
    print(f'最终测试集准确率: {best_accuracy:.2f}%')

    # 加载最佳模型权重并保存
    model.load_state_dict(best_model_weights)
    if save_model:
        torch.save(model.state_dict(), model_path)
        print(f'最佳模型已保存至: {model_path}')

    # 绘制训练曲线
    set_chinese_font()
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
    plt.title('训练损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失值')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-')
    plt.title('测试集准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')

    plt.tight_layout()
    plt.show()

    return total_time
