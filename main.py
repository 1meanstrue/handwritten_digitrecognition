import os
import torch
import time
from train import train_model
from predict import (
    batch_validate_test_dataset,
    visualize_error_cases
)
from utils import set_chinese_font, plot_speed_comparison

# 初始化中文显示
set_chinese_font()


def compare_cpu_gpu_training(model_type='CNN', epochs=10, batch_size=64):
    """对比CPU和GPU的完整训练过程速率"""
    # 确保模型文件不存在（避免直接加载预训练模型，确保重新训练）
    model_path = f'mnist_{model_type.lower()}_model.pth'
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"已删除现有模型{model_path}，将重新训练")

    # 1. CPU训练
    print("\n===== 开始CPU训练 =====")
    cpu_device = torch.device('cpu')
    cpu_time = train_model(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        device=cpu_device,
        save_model=True
    )

    # 2. GPU训练（如果可用）
    gpu_time = None
    if torch.cuda.is_available():
        # 删除CPU训练的模型，确保GPU重新训练
        if os.path.exists(model_path):
            os.remove(model_path)

        print("\n===== 开始GPU训练 =====")
        gpu_device = torch.device('cuda')
        gpu_time = train_model(
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            device=gpu_device,
            save_model=True
        )
    else:
        print("\nGPU不可用，跳过GPU训练对比")

    # 3. 显示对比结果
    if gpu_time is not None:
        print("\n===== CPU vs GPU 训练速率对比 =====")
        print(f"CPU总训练耗时: {cpu_time:.2f}秒")
        print(f"GPU总训练耗时: {gpu_time:.2f}秒")
        print(f"GPU比CPU快: {cpu_time / gpu_time:.2f}倍")

        # 可视化对比
        plot_speed_comparison(cpu_time, gpu_time, "完整训练过程")

    return cpu_time, gpu_time


def main():
    # 配置
    model_type = 'ImprovedCNN'  # 使用改进的模型
    epochs = 30  # 训练轮次
    batch_size = 128  # 批次大小

    # 1. 对比CPU和GPU的完整训练速率
    print("=== 开始CPU与GPU训练速率对比 ===")
    cpu_time, gpu_time = compare_cpu_gpu_training(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size
    )

    # 2. 使用最终训练好的模型进行批量验证（全部10000张图片）
    model_path = f'mnist_{model_type.lower()}_model.pth'
    if os.path.exists(model_path):
        print(f"\n=== 使用{model_type}模型批量验证MNIST测试集 ===")
        batch_accuracy, error_cases = batch_validate_test_dataset(
            model_path=model_path,
            model_type=model_type,
            sample_size=10000,  # 验证全部10000张图片
            random_seed=42,
            save_errors=True
        )

        # 3. 可视化错误案例 - 修复变量未定义问题
        if error_cases:
            print(f"\n=== 可视化错误案例 ===")
            num_to_show = 100  # 定义要显示的错误案例数量
            visualize_error_cases(error_cases, num_to_show)


if __name__ == "__main__":
    main()
