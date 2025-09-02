import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os

def train_model(model, train_loader, epochs=10, learning_rate=0.001, device=None):
    """
    训练模型
    
    Args:
        model (nn.Module): 要训练的模型
        train_loader (DataLoader): 训练数据加载器
        epochs (int): 训练轮数
        learning_rate (float): 学习率
        device (torch.device): 训练设备
        
    Returns:
        model (nn.Module): 训练好的模型
    """
    # 如果没有指定设备，则自动选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将模型移动到指定设备
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 定义学习率调度器
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练循环
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据移动到指定设备
            data, target = data.to(device), target.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 打印训练进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # 计算并打印每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs} - 平均损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
        
        # 更新学习率
        scheduler.step()
    
    return model

def save_model(model, filepath):
    """
    保存模型
    
    Args:
        model (nn.Module): 要保存的模型
        filepath (str): 保存路径
    """
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存到: {filepath}")

if __name__ == "__main__":
    # 这里只是示例代码，实际使用时需要导入其他模块
    print("训练模块已定义")
    print("请通过主程序调用train_model函数进行训练")