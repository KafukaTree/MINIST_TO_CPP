import torch
import torch.nn as nn

def evaluate_model(model, test_loader, device=None):
    """
    在测试集上评估模型性能
    
    Args:
        model (nn.Module): 要评估的模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 评估设备
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    # 如果没有指定设备，则自动选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将模型移动到指定设备并设置为评估模式
    model.to(device)
    model.eval()
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 统计变量
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 禁用梯度计算
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移动到指定设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            test_loss += criterion(output, target).item()
            
            # 计算准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # 计算平均损失和准确率
    average_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return average_loss, accuracy

def print_evaluation_results(average_loss, accuracy):
    """
    打印评估结果
    
    Args:
        average_loss (float): 平均损失
        accuracy (float): 准确率
    """
    print(f"测试集平均损失: {average_loss:.4f}")
    print(f"测试集准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    # 这里只是示例代码，实际使用时需要导入其他模块
    print("评估模块已定义")
    print("请通过主程序调用evaluate_model函数进行评估")