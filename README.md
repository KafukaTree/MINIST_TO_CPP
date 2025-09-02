# MNIST 手写数字识别 LibTorch C++ 部署方案

这是一个使用 LibTorch (PyTorch C++ API) 实现的轻量级部署方案，用于识别手写数字图像。

## 项目结构

```
.
├── CMakeLists.txt          # CMake 构建配置文件
├── README.md               # 项目说明文档
├── mnist_resnet18.pth      # PyTorch 模型权重文件
├── mnist_resnet18_traced.pt # TorchScript 模型文件
├── src/
│   └── main.cpp            # 主程序源代码
└── build/                  # 构建目录（需手动创建）
```

## 环境要求

- C++17 或更高版本
- LibTorch 1.7.0 或更高版本
- OpenCV 4.x
- CMake 3.10 或更高版本

## 构建步骤

1. 下载并安装 LibTorch:
   - 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 下载 LibTorch
   - 解压到合适的位置，例如 `C:\libtorch`

2. 安装 OpenCV:
   - 下载并安装 OpenCV 或使用包管理器安装

3. 创建构建目录并进入:
   ```bash
   mkdir build
   cd build
   ```

4. 配置 CMake (Windows 示例):
   ```bash
   cmake -DCMAKE_PREFIX_PATH=C:\libtorch ..  # 替换为你的 LibTorch 路径
   ```

5. 构建项目:
   ```bash
   cmake --build . --config Release
   ```

## 运行应用

构建完成后，可执行文件将在 `build/Release/` 目录下（Windows）或 `build/` 目录下（Linux/macOS）。

运行应用需要提供模型文件和待识别图像的路径：

```bash
./mnist_classifier ../mnist_resnet18_traced.pt <image_path>
```

## Python 脚本

- `convert_to_torchscript.py`: 将 PyTorch 模型转换为 TorchScript 格式的脚本

## 注意事项

1. 输入图像应为灰度图像
2. 图像会被自动调整为 28x28 像素大小
3. 模型期望输入为单通道图像

## 性能特点

- 轻量级部署：不依赖 Python 环境
- 快速推理：利用 LibTorch 优化的推理引擎
- 跨平台：可在 Windows、Linux 和 macOS 上运行