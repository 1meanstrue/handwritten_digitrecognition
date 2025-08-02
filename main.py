# ============================ 1. 导入库与环境设置 ==========================
'''
PyTorch 相关库：用于构建深度学习模型、数据加载和处理
PIL 库：用于图像处理和操作
Matplotlib 库：用于可视化结果和调试过程
OS 库：用于文件系统操作
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

'''
字体设置：确保 Matplotlib 能够正确显示中文
设备检测：自动检测是否有可用的 GPU，优先使用 GPU 进行计算
'''
# 设置中文字体（解决中文显示警告）
font_path = fm.findfont(fm.FontProperties(family=['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']))
plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
# 检测设备（自动选择 GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ========================== 2. 定义模型（CNN) ==========================
'''
网络结构：包含两层卷积层和两层全连接层，适合处理 MNIST 手写数字数据
关键操作：
卷积层：提取图像特征
最大池化：减小特征图尺寸，增加感受野
ReLU 激活函数：引入非线性
Dropout：随机失活神经元，防止过拟合
全连接层：将特征映射到 10 个数字类别
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 28→12
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))  # 12→4
        x = x.view(-1, 320)  # 展平为向量
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

'''
简单模型：仅包含两层全连接层，不使用卷积操作
用途：用于对比卷积网络和简单全连接网络的性能差异
'''
class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平为一维向量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ======================== 3. 增强预处理（带可视化调试） ========================
def preprocess_image(image_path, debug=False):
    """
    增强预处理，确保输出与MNIST一致：
    1. 裁剪边缘空白  2. 缩放到20×20  3. 填充到28×28居中
    4. 二值化        5. 反色（黑底白字）  6. 归一化
    debug=True时，可视化每一步结果
    """
    # 1. 打开图像并转灰度
    try:
        img = Image.open(image_path).convert('L')
    except Exception as e:
        raise FileNotFoundError(f"无法打开图像: {e}")

    if debug:
        plt.imshow(img, cmap='gray')
        plt.title("步骤1：原始灰度图像")
        plt.axis('off')
        plt.show()

    # 2. 裁剪边缘空白（去除多余白底）
    bbox = img.getbbox()
    if bbox:  # 如果有内容才裁剪
        img = img.crop(bbox)
    else:
        raise ValueError("图像全空白，无法识别")

    if debug:
        plt.imshow(img, cmap='gray')
        plt.title("步骤2：裁剪边缘后")
        plt.axis('off')
        plt.show()

    # 3. 缩放到20×20（MNIST数字的典型大小）
    img = img.resize((20, 20), Image.LANCZOS)  # 高质量缩放

    if debug:
        plt.imshow(img, cmap='gray')
        plt.title("步骤3：缩放到20×20后")
        plt.axis('off')
        plt.show()

    # 4. 填充到28×28并居中（模拟MNIST格式）
    new_img = Image.new('L', (28, 28), 255)  # 28×28 白色背景
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    new_img.paste(img, (x_offset, y_offset))

    if debug:
        plt.imshow(new_img, cmap='gray')
        plt.title("步骤4：填充到28×28后")
        plt.axis('off')
        plt.show()

    # 5. 二值化（强化黑白对比）
    threshold = 180  # 可调整阈值（0-255）
    new_img = new_img.point(lambda p: 0 if p < threshold else 255)

    if debug:
        plt.imshow(new_img, cmap='gray')
        plt.title(f"步骤5：二值化（阈值{threshold}）")
        plt.axis('off')
        plt.show()

    # 6. 反色（转为黑底白字，与MNIST一致）
    # 动态判断是否需要反色：如果图像大部分是白色（背景），则反色
    pixel_sum = sum(new_img.getdata())
    total_pixels = 28 * 28
    if pixel_sum / total_pixels > 128:  # 背景是白色，需要反色
        new_img = ImageOps.invert(new_img)
        if debug:
            plt.imshow(new_img, cmap='gray')
            plt.title("步骤6：反色（黑底白字）")
            plt.axis('off')
            plt.show()
    else:
        if debug:
            print("已是黑底白字，无需反色")

    # 7. 转为张量并归一化（使用MNIST的均值和标准差）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(new_img).unsqueeze(0)  # 添加batch维度 [1,1,28,28]

    return img_tensor, new_img
'''
预处理流程：
灰度转换：将彩色图像转为灰度图
边缘裁剪：去除图像周围的空白区域
尺寸调整：将数字区域缩放到 20×20 像素
居中填充：填充到 28×28 像素，与 MNIST 格式一致
二值化：将图像转为黑白两色，增强对比度
反色处理：确保数字为白色，背景为黑色
张量转换：转为 PyTorch 张量并归一化
调试模式：当debug=True时，会可视化每一步预处理结果
'''


# ======================== 4. 预测函数（带调试信息） ========================
def predict_digit(image_path, model_path='mnist_cnn_model.pth', debug=False):
    """
    加载模型并预测，debug=True时打印张量信息
    """
    # 加载模型（自动适配设备）
    model = CNN()  # 或替换为 SimpleFC() 测试
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 预处理（带调试可视化）
    img_tensor, processed_img = preprocess_image(image_path, debug=debug)
    img_tensor = img_tensor.to(device)

    # 调试：打印张量统计信息
    if debug:
        print(f"输入张量形状: {img_tensor.shape}")
        print(f"最小值: {torch.min(img_tensor).item():.4f}")
        print(f"最大值: {torch.max(img_tensor).item():.4f}")
        print(f"均值: {torch.mean(img_tensor).item():.4f}")

    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0] * 100  # 转换为百分比
        predicted_class = output.argmax(1).item()
        confidence = probabilities[predicted_class].item()

    return predicted_class, confidence, processed_img, probabilities
'''
预测流程：
加载预训练模型
对输入图像进行预处理
将图像张量传入模型进行推理
使用 softmax 函数计算各数字类别的概率
返回预测结果、置信度和概率分布
'''

# ======================== 4. 可视化识别结果 ========================
def recognize_handwritten_digit(image_path, debug=False):
    """
    完整识别流程：显示原始图、处理后图、概率分布
    debug=True时开启预处理调试和张量信息打印
    """
    try:
        # 预测（带调试）
        digit, confidence, processed_img, probabilities = predict_digit(image_path, debug=debug)

        # 显示原始图像
        original_img = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("原始图像")
        plt.axis('off')

        # 显示处理后图像
        plt.subplot(1, 2, 2)
        plt.imshow(processed_img, cmap='gray')
        plt.title(f"处理后图像\n预测: {digit}, 置信度: {confidence:.2f}%")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # 显示概率分布
        plt.figure(figsize=(10, 4))
        digits = list(range(10))
        plt.bar(digits, probabilities.cpu().numpy(), color='skyblue')
        plt.xticks(digits)
        plt.xlabel('数字')
        plt.ylabel('概率 (%)')
        plt.title('各数字预测概率')
        plt.tight_layout()
        plt.show()

        # 打印结果
        print(f"识别结果: 数字 {digit}")
        print(f"置信度: {confidence:.2f}%")
        print("\n各数字概率分布:")
        for i in range(10):
            print(f"数字 {i}: {probabilities[i]:.2f}%")

    except Exception as e:
        print(f"识别失败: {e}")
'''
可视化功能：
显示原始输入图像
显示预处理后的图像及预测结果
绘制各数字类别的概率分布柱状图
打印详细的识别结果和置信度
'''

# ======================== 5. 训练模型（首次运行需训练） ========================
def train_model(model_type='CNN', epochs=15, batch_size=64):
    """
    自动下载MNIST并训练，支持 'CNN' 或 'SimpleFC'
    """
    # 数据增强（提升模型鲁棒性）
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    print("正在下载MNIST数据集...")
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    # 初始化模型
    if model_type == 'CNN':
        model = CNN().to(device)
    elif model_type == 'SimpleFC':
        model = SimpleFC().to(device)
    else:
        raise ValueError("model_type 必须是 'CNN' 或 'SimpleFC'")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print(f"开始训练 {model_type} 模型，共 {epochs} 轮...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每100批次打印进度
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | 批次 {batch_idx}/{len(train_loader)} | 损失: {loss.item():.4f}")

        # 测试集验证
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} | 测试损失: {test_loss:.4f} | 准确率: {accuracy:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), f'mnist_{model_type.lower()}_model.pth')
    print(f"模型已保存为 mnist_{model_type.lower()}_model.pth")
'''
训练流程：
数据增强：对训练数据应用随机旋转、平移和缩放，提升模型鲁棒性
数据集加载：自动下载 MNIST 数据集
模型初始化：根据参数选择 CNN 或 SimpleFC 模型
优化器设置：使用 Adam 优化器，学习率为 0.001
损失函数：使用交叉熵损失函数，适合多分类任务
训练循环：迭代训练多个轮次，每轮次包含前向传播、损失计算、反向传播和参数更新
模型评估：在测试集上评估模型性能，计算损失和准确率
模型保存：保存训练好的模型参数
'''

# ======================== 6. 主程序（自动判断是否训练） ========================
def main():
    # 检查是否需要训练模型
    model_path = 'mnist_cnn_model.pth'  # 或 'mnist_simplefc_model.pth'
    if not os.path.exists(model_path):
        print("未检测到预训练模型，开始训练...")
        train_model(model_type='CNN', epochs=15)  # 首次训练建议用CNN

    # 替换为你的图片路径
    image_path = r"test number\0_image.jpg"

    # 执行识别（debug=True 可开启预处理调试）
    print(f"开始识别图像: {image_path}")
    recognize_handwritten_digit(image_path, debug=True)  # 调试时开启debug


if __name__ == "__main__":
    main()
'''
主程序逻辑：
检查是否存在预训练模型，不存在则自动训练
设置待识别的图像路径
调用识别函数，开启调试模式（可选）
'''