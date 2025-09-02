# 手写数字识别应用改进与MNIST模型训练

## 项目概述

这是一个基于 OpenCV 和 LibTorch 的手写数字识别应用，用户可以在界面上手写数字，应用会实时识别并显示结果。本文档描述了对原应用的改进以及用于训练MNIST识别模型的Python代码。

## 功能改进

### 1. 识别结果清除逻辑优化

在原始实现中，当用户按下'r'键进行识别时，上一次的识别结果不会被清除，这会影响用户体验。我们对代码进行了以下改进：

#### DrawingCanvas类的改进

1. **完善clear()方法**：
   ```cpp
   void DrawingCanvas::clear()
   {
       canvas = cv::Scalar(0, 0, 0);  // Reset to black background
       resultText = "Draw a digit and press 'r' to recognize";
   }
   ```

2. **优化'r'键处理逻辑**：
   ```cpp
   } else if (key == 'r' || key == 'R') {  // R key to recognize
       recognizeFlag = true;
       resultText = "Recognizing...";  // Clear previous result and show recognizing message
       break;
   }
   ```

3. **自动清除画布**：
   在main.cpp中，识别完成后自动清除画布：
   ```cpp
   // Clear canvas for next drawing
   canvas.clear();
   ```

这些改进确保了：
- 按下'r'键时，上一次的识别结果会被清除
- 识别过程中显示"Recognizing..."提示
- 识别完成后自动清除画布，为下一次绘制做准备

### 2. 用户体验提升

- 优化了界面文字显示，增大了字体大小和画布尺寸
- 改进了界面布局，使操作说明更加清晰
- 提供了更好的视觉反馈

## MNIST模型训练

### 项目结构

```
.
├── data.py              # 数据加载和预处理
├── model.py             # 模型定义（基于ResNet18）
├── train.py             # 训练逻辑
├── eval.py              # 评估逻辑
├── main.py              # 训练主程序
├── convert_to_torchscript.py  # 模型转换为TorchScript格式
└── requirements.txt     # 依赖包列表
```

### 数据处理 (data.py)

使用PyTorch的DataLoader加载MNIST数据集，并进行标准化处理：

```python
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
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
```

### 模型定义 (model.py)

使用ResNet18作为基础模型，并进行适配以处理MNIST数据集：

```python
class MNISTResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTResNet18, self).__init__()
        
        # 加载预训练的ResNet18模型
        self.resnet18 = models.resnet18(pretrained=False)
        
        # 修改第一层卷积层以适应单通道输入（MNIST是灰度图像）
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 修改最后的全连接层以适应MNIST的类别数
        self.resnet18.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet18(x)
```

### 训练过程 (train.py)

实现了一个完整的训练循环，包括损失计算、反向传播和参数更新：

```python
def train_model(model, train_loader, epochs=10, learning_rate=0.001, device=None):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 定义学习率调度器
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练循环
    model.train()
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
```

### 模型评估 (eval.py)

在测试集上评估模型性能：

```python
def evaluate_model(model, test_loader, device=None):
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
```

### 模型转换 (convert_to_torchscript.py)

将训练好的PyTorch模型转换为TorchScript格式，以便在C++应用中使用：

```python
def convert_model_to_torchscript():
    # 创建模型实例
    model = get_model(num_classes=10)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load("mnist_resnet18.pth", map_location=torch.device('cpu')))
    
    # 设置为评估模式
    model.eval()
    
    # 创建示例输入
    example_input = torch.randn(1, 1, 28, 28)
    
    # 将模型转换为TorchScript格式
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存TorchScript模型
    traced_model.save("mnist_resnet18_traced.pt")
```

### 训练主程序 (main.py)

整合所有组件的训练主程序：

```python
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    print("加载数据...")
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # 获取模型
    print("创建模型...")
    model = get_model(num_classes=10)
    
    # 训练模型
    print("开始训练...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        epochs=5,
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
```

## 使用说明

### 训练模型

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行训练：
   ```bash
   python main.py
   ```

3. 转换模型为TorchScript格式：
   ```bash
   python convert_to_torchscript.py
   ```

### 构建和运行C++应用

1. 确保已安装 OpenCV 和 LibTorch

2. 创建构建目录并进入:
   ```bash
   mkdir build
   cd build
   ```

3. 配置 CMake:
   ```bash
   cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch" ..
   ```
   
   Windows 示例:
   ```bash
   cmake -DCMAKE_PREFIX_PATH="C:\libtorch" ..
   ```

4. 构建项目:
   ```bash
   cmake --build . --config Release
   ```

5. 运行应用:
   ```bash
   ./handwriting_recognizer_opencv
   ```

## 界面操作

- **鼠标左键拖拽**: 在画板上绘制数字
- **'r' 或 'R' 键**: 识别当前绘制的数字
- **'c' 或 'C' 键**: 清除画板
- **ESC 键**: 退出应用

## 改进效果

通过以上改进，应用的用户体验得到了显著提升：
1. 识别结果的清除逻辑更加合理
2. 界面交互更加流畅
3. 为下一次绘制自动准备画布