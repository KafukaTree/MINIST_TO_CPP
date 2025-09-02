import torch
import torch.nn as nn
from torchvision import models

class MNISTResNet18(nn.Module):
    def __init__(self, num_classes=10):
        """
        初始化MNIST适配的ResNet18模型
        
        Args:
            num_classes (int): 分类数量，默认为10（MNIST数据集的类别数）
        """
        super(MNISTResNet18, self).__init__()
        
        # 加载预训练的ResNet18模型
        self.resnet18 = models.resnet18(pretrained=False)
        
        # 修改第一层卷积层以适应单通道输入（MNIST是灰度图像）
        # 原始ResNet18的第一层是Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # 我们将其修改为适应单通道输入
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 修改最后的全连接层以适应MNIST的类别数
        # 原始ResNet18的全连接层是Linear(512, 1000)
        # 我们将其修改为Linear(512, num_classes)
        self.resnet18.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 输出张量
        """
        return self.resnet18(x)

def get_model(num_classes=10):
    """
    获取MNIST适配的ResNet18模型实例
    
    Args:
        num_classes (int): 分类数量
        
    Returns:
        MNISTResNet18: 模型实例
    """
    return MNISTResNet18(num_classes)

if __name__ == "__main__":
    # 测试模型
    model = get_model()
    print("模型结构:")
    print(model)
    
    # 创建一个示例输入（批次大小为4，单通道，28x28图像）
    sample_input = torch.randn(4, 1, 28, 28)
    output = model(sample_input)
    print(f"\n输入张量形状: {sample_input.shape}")
    print(f"输出张量形状: {output.shape}")