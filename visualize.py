import torch
import matplotlib.pyplot as plt
import numpy as np
from data import get_data_loaders
from model import get_model
# 中文展示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


def visualize_samples(model=None, num_samples=10, device=None):
    """
    可视化MNIST数据集中的样本
    
    Args:
        model (nn.Module): 训练好的模型（可选）
        num_samples (int): 要展示的样本数量
        device (torch.device): 设备
    """
    # 如果没有指定设备，则自动选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取测试数据加载器
    _, test_loader = get_data_loaders(batch_size=1)
    
    # 将模型移动到指定设备（如果有提供）
    if model is not None:
        model.to(device)
        model.eval()
    
    # 创建图表
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    # 获取一个批次的数据迭代器
    data_iter = iter(test_loader)
    
    for i in range(num_samples):
        # 获取下一个样本
        try:
            image, label = next(data_iter)
        except StopIteration:
            # 如果数据用完了，重新开始
            data_iter = iter(test_loader)
            image, label = next(data_iter)
        
        # 将图像数据转换为numpy数组用于显示
        img = image.squeeze().numpy()
        
        # 如果有模型，进行预测
        if model is not None:
            with torch.no_grad():
                image = image.to(device)
                output = model(image)
                pred = output.argmax(dim=1, keepdim=True)
                pred_label = pred.item()
                confidence = torch.softmax(output, dim=1).max().item()
                
                # 设置标题（真实标签和预测标签）
                title = f"真实: {label.item()}, 预测: {pred_label}\n置信度: {confidence:.2f}"
                color = "green" if label.item() == pred_label else "red"
        else:
            # 只显示真实标签
            title = f"真实标签: {label.item()}"
            color = "black"
        
        # 显示图像
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def load_trained_model(model_path="mnist_resnet18.pth", num_classes=10, device=None):
    """
    加载训练好的模型
    
    Args:
        model_path (str): 模型文件路径
        num_classes (int): 分类数量
        device (torch.device): 设备
        
    Returns:
        nn.Module: 加载的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例
    model = get_model(num_classes)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 设置为评估模式
    model.eval()
    
    print(f"模型已从 {model_path} 加载")
    return model

if __name__ == "__main__":
    # 只显示样本（不进行预测）
    print("显示MNIST数据集样本...")
    visualize_samples()
    
    # 如果存在训练好的模型，加载并进行预测展示
    import os
    if os.path.exists("mnist_resnet18.pth"):
        print("\n加载训练好的模型并进行预测展示...")
        model = load_trained_model()
        visualize_samples(model)
    else:
        print("\n未找到训练好的模型文件，只显示样本标签")