import torch
import os
import re
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def set_chinese_font():
    """设置matplotlib中文显示"""
    font_path = fm.findfont(fm.FontProperties(family=['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']))
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()


def get_device():
    """获取计算设备（优先GPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timeit(func):
    """装饰器：计算函数运行时间"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start  # 返回结果和耗时（秒）
    return wrapper


def extract_label_from_filename(filename):
    """从文件名提取标签（假设格式：{label}_image.jpg）"""
    match = re.match(r'^(\d+)_image\.jpg$', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"文件名格式错误：{filename}（需为{label}_image.jpg）")


def plot_accuracy_curve(epochs, acc_list, model_type):
    """绘制准确率曲线"""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), acc_list, marker='o', color='b')
    plt.xlabel('训练轮次（Epoch）')
    plt.ylabel('准确率（%）')
    plt.title(f'{model_type}模型训练准确率曲线')
    plt.grid(True)
    plt.savefig(f'{model_type}_accuracy_curve.png')
    plt.show()


def plot_speed_comparison(cpu_time, gpu_time, task_name):
    """绘制CPU/GPU速率对比图"""
    plt.figure(figsize=(8, 5))
    plt.bar(['CPU', 'GPU'], [cpu_time, gpu_time], color=['gray', 'green'])
    plt.ylabel('耗时（秒）')
    plt.title(f'CPU与GPU{task_name}速率对比')
    for i, v in enumerate([cpu_time, gpu_time]):
        plt.text(i, v+0.01, f'{v:.4f}s', ha='center')
    plt.savefig(f'{task_name}_speed_comparison.png')
    plt.show()