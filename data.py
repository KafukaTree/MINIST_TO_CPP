import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    """
    获取MNIST数据集的训练和测试数据加载器
    
    Args:
        batch_size (int): 批次大小
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载训练数据集
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 加载测试数据集
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # 测试数据加载器
    train_loader, test_loader = get_data_loaders()
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    
    # 显示一个批次的数据信息
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"图像张量形状: {images.shape}")
    print(f"标签张量形状: {labels.shape}")