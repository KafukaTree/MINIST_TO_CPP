import torch
from model import get_model

def convert_model_to_torchscript():
    """
    将训练好的PyTorch模型转换为TorchScript格式
    """
    # 创建模型实例
    model = get_model(num_classes=10)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load("mnist_resnet18.pth", map_location=torch.device('cpu')))
    
    # 设置为评估模式
    model.eval()
    
    # 创建示例输入（批次大小为1，单通道，28x28图像）
    example_input = torch.randn(1, 1, 28, 28)
    
    # 将模型转换为TorchScript格式
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存TorchScript模型
    traced_model.save("mnist_resnet18_traced.pt")
    print("模型已成功转换为TorchScript格式并保存为 mnist_resnet18_traced.pt")
    
    # 验证转换后的模型
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = traced_model(example_input)
        print(f"原始模型输出: {original_output}")
        print(f"TorchScript模型输出: {traced_output}")
        print(f"输出差异: {torch.max(torch.abs(original_output - traced_output))}")

if __name__ == "__main__":
    convert_model_to_torchscript()