import torch
import argparse
import os

# 导入自定义模块
from data import get_data_loaders
from model import get_model
from train import train_model, save_model
from eval import evaluate_model, print_evaluation_results

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    print("加载数据...")
    train_loader, test_loader = get_data_loaders(batch_size=64)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 获取模型
    print("创建模型...")
    model = get_model(num_classes=10)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("开始训练...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        epochs=5,  # 为了快速演示，只训练5个epoch
        learning_rate=0.001,
        device=device
    )
    
    # 保存模型
    save_model(trained_model, "mnist_resnet18.pth")
    
    # 评估模型
    print("评估模型...")
    avg_loss, accuracy = evaluate_model(trained_model, test_loader, device)
    print_evaluation_results(avg_loss, accuracy)
    
    print("训练和评估完成!")

if __name__ == "__main__":
    main()