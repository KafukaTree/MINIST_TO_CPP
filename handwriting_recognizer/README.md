# 手写数字识别 Qt 应用

这是一个基于 Qt 和 LibTorch 的手写数字识别应用，用户可以在界面上手写数字，应用会实时识别并显示结果。

## 功能特点

- 手写数字绘制画板
- 实时数字识别
- 置信度显示
- 清除画板功能
- 轻量级部署方案

## 项目结构

```
handwriting_recognizer/
├── CMakeLists.txt              # CMake 构建配置文件
├── README.md                   # 项目说明文档
├── main.cpp                    # 应用入口点
├── mainwindow.h/cpp            # 主窗口类
├── drawingwidget.h/cpp         # 手写画板控件
├── torchmodel.h/cpp            # LibTorch 模型处理类
└── build/                      # 构建目录
```

## 环境要求

- C++17 或更高版本
- Qt 5.15 或 Qt 6.x
- LibTorch 1.7.0 或更高版本
- OpenCV 4.x
- CMake 3.16 或更高版本

## 构建步骤

1. 确保已安装 Qt、LibTorch 和 OpenCV

2. 创建构建目录并进入:
   ```bash
   mkdir build
   cd build
   ```

3. 配置 CMake:
   ```bash
   cmake -DCMAKE_PREFIX_PATH="/path/to/Qt;/path/to/libtorch" ..
   ```
   
   Windows 示例:
   ```bash
   cmake -DCMAKE_PREFIX_PATH="C:\Qt\5.15.2\msvc2019_64;C:\libtorch" ..
   ```

4. 构建项目:
   ```bash
   cmake --build . --config Release
   ```

## 运行应用

构建完成后，可执行文件将在 `build/bin/` 目录下。

运行应用前，请确保以下文件在可执行文件同目录下：
- `mnist_resnet18_traced.pt` (TorchScript 模型文件)

```bash
./handwriting_recognizer
```

## 使用说明

1. 在黑色画板区域用鼠标左键绘制数字
2. 点击"识别"按钮进行数字识别
3. 查看识别结果和置信度
4. 点击"清除"按钮清空画板

## 部署方案特点

- 轻量级: 不依赖 Python 环境
- 高性能: 利用 LibTorch 优化的推理引擎
- 跨平台: 可在 Windows、Linux 和 macOS 上运行
- 易部署: 只需可执行文件、模型文件和必要的动态库