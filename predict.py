import torch
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from models import CNN, SimpleFC, ImprovedCNN
from utils import get_device

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# MNIST数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])


def predict_digit(image_path, model_path='mnist_cnn_model.pth', model_type='CNN', debug=False):
    """预测单张图片的数字"""
    device = get_device()

    # 加载模型
    if model_type == 'CNN':
        model = CNN()
    elif model_type == 'SimpleFC':
        model = SimpleFC()
    elif model_type == 'ImprovedCNN':
        model = ImprovedCNN()
    else:
        raise ValueError("model_type必须是'CNN'、'SimpleFC'或'ImprovedCNN'")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 预处理图片
    img = Image.open(image_path).convert('L')  # 转为灰度图
    img = img.resize((28, 28))  # 调整为MNIST尺寸
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # 转为张量并添加批次维度
    img_tensor = (1 - img_tensor)  # 反转颜色（MNIST是白字黑底）
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)  # 标准化

    processed_img = img_tensor.squeeze().numpy()  # 用于可视化

    if debug:
        print(f"输入张量形状: {img_tensor.shape}")
        print(f"最小值: {torch.min(img_tensor).item():.4f}")
        print(f"最大值: {torch.max(img_tensor).item():.4f}")
        print(f"均值: {torch.mean(img_tensor).item():.4f}")

    with torch.no_grad():
        output = model(img_tensor.to(device))
        probabilities = F.softmax(output, dim=1)[0] * 100
        predicted_class = output.argmax(1).item()
        confidence = probabilities[predicted_class].item()

    return predicted_class, confidence, processed_img, probabilities


def visualize_result(image_path, predicted_class, confidence, processed_img, probabilities):
    """可视化单张图片的预测结果"""
    original_img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("原始图像")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title(f"处理后图像\n预测: {predicted_class}, 置信度: {confidence:.2f}%")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    digits = list(range(10))
    plt.bar(digits, probabilities.cpu().numpy(), color='skyblue')
    plt.xticks(digits)
    plt.xlabel('数字')
    plt.ylabel('概率 (%)')
    plt.title('各数字预测概率分布')
    plt.tight_layout()
    plt.show()


def batch_validate_test_dataset(
        model_path='mnist_cnn_model.pth',
        model_type='CNN',
        sample_size=10000,  # 默认验证全部10000张图片
        random_seed=42,
        save_errors=False,
        error_save_dir='error_samples'
):
    """批量验证MNIST官方测试集，收集错误案例并按样本索引升序排序"""
    correct = 0
    total = 0
    error_cases = []  # 存储格式: (索引, 真实标签, 预测结果, 置信度, 图像数据, 概率分布)

    # 创建错误样本保存目录
    if save_errors and not os.path.exists(error_save_dir):
        os.makedirs(error_save_dir)

    # 加载MNIST测试集
    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    dataset_size = len(test_dataset)  # 固定为10000

    # 设置随机种子确保可复现
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # 选择验证样本
    if sample_size is not None and dataset_size > sample_size:
        indices = torch.randperm(dataset_size)[:sample_size]
        print(f"从{dataset_size}个测试样本中随机选取{sample_size}个进行验证...")
    else:
        indices = torch.arange(dataset_size)  # 使用全部样本
        print(f"使用全部{dataset_size}个测试样本进行验证...")

    # 加载模型
    device = get_device()
    if model_type == 'CNN':
        model = CNN()
    elif model_type == 'SimpleFC':
        model = SimpleFC()
    elif model_type == 'ImprovedCNN':
        model = ImprovedCNN()
    else:
        raise ValueError("model_type必须是'CNN'、'SimpleFC'或'ImprovedCNN'")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 批量预测
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                image, true_label = test_dataset[idx]
                image_tensor = image.unsqueeze(0).to(device)  # 添加批次维度

                # 模型预测
                output = model(image_tensor)
                probabilities = F.softmax(output, dim=1)[0] * 100
                predicted_class = output.argmax(1).item()
                confidence = probabilities[predicted_class].item()

                # 统计结果
                total += 1
                if predicted_class == true_label:
                    correct += 1
                else:
                    # 保存错误案例详情
                    error_cases.append((
                        idx.item(),  # 样本索引
                        true_label,  # 真实标签
                        predicted_class,  # 预测结果
                        confidence,  # 置信度
                        image_tensor.cpu(),  # 图像数据
                        probabilities.cpu()  # 概率分布
                    ))

                    # 保存错误样本图像
                    if save_errors:
                        img_np = image.squeeze().numpy()
                        plt.imsave(
                            os.path.join(error_save_dir,
                                         f"error_idx{idx.item()}_true{true_label}_pred{predicted_class}.png"),
                            img_np,
                            cmap='gray'
                        )

                # 打印进度（每1000个样本更新一次）
                if total % 1000 == 0:
                    print(f"已处理{total}/{len(indices)}个样本 | 当前准确率: {(correct / total) * 100:.2f}%")

            except Exception as e:
                print(f"处理样本索引{idx}失败: {e}")

    # 按样本索引升序排序错误案例（核心修改）
    error_cases.sort(key=lambda x: x[0])

    # 计算并显示准确率
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n批量验证结果：总样本数 {total} | 正确 {correct} | 准确率 {accuracy:.2f}%")

    # 错误案例分析（已按索引排序）
    if error_cases:
        print(f"\n===== 错误案例分析（按样本索引升序） =====")
        print(f"总错误数: {len(error_cases)}")

        # 统计错误分布
        error_distribution = {}
        for case in error_cases:
            true_label = case[1]
            error_distribution[true_label] = error_distribution.get(true_label, 0) + 1

        # 打印错误分布
        print("数字错误分布:")
        for digit in sorted(error_distribution.keys()):
            print(f"数字 {digit}: {error_distribution[digit]} 次错误")

        # 显示前5个错误案例（按索引排序后的结果）
        print(f"\n错误案例详情:")
        for case in error_cases[:]:
            print(f"样本索引: {case[0]} | 真实: {case[1]} | 预测: {case[2]} | 置信度: {case[3]:.2f}%")

    return accuracy, error_cases


def visualize_error_cases(error_cases, num_to_show):
    """可视化错误样本及预测概率分布（按索引升序展示）"""
    if not error_cases:
        print("没有错误案例可可视化")
        return

    # 因已排序，直接取前N个就是索引最小的N个
    cases_to_show = error_cases[:num_to_show]

    for i, case in enumerate(cases_to_show):
        idx, true_label, pred_label, confidence, image, probabilities = case

        plt.figure(figsize=(10, 4))
        plt.suptitle(f"错误案例 {i + 1}/{len(cases_to_show)} | 样本索引: {idx}", fontsize=12)

        # 显示图像
        plt.subplot(1, 2, 1)
        img = image.squeeze(0).squeeze(0).numpy()  # 移除批次和通道维度
        plt.imshow(img, cmap='gray')
        plt.title(f"真实: {true_label} | 预测: {pred_label}\n置信度: {confidence:.2f}%")
        plt.axis('off')

        # 显示概率分布
        plt.subplot(1, 2, 2)
        digits = list(range(10))
        bars = plt.bar(digits, probabilities.numpy(), color='salmon')
        bars[true_label].set_color('green')  # 真实标签为绿色
        bars[pred_label].set_color('red')  # 预测标签为红色
        plt.xticks(digits)
        plt.xlabel('数字')
        plt.ylabel('概率 (%)')
        plt.title('预测概率分布')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 避免标题重叠
        plt.show()


def visualize_result_from_dataset(sample_data, predicted_class, confidence, probabilities):
    """可视化数据集中的样本预测结果"""
    image = sample_data.squeeze(0).squeeze(0).numpy()  # 处理张量维度

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"MNIST测试样本\n预测: {predicted_class}, 置信度: {confidence:.2f}%")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    digits = list(range(10))
    plt.bar(digits, probabilities.cpu().numpy(), color='skyblue')
    plt.xticks(digits)
    plt.xlabel('数字')
    plt.ylabel('概率 (%)')
    plt.title('各数字预测概率分布')
    plt.tight_layout()
    plt.show()
