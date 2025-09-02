# 手写数字识别 OpenCV 应用

这是一个基于 OpenCV 和 LibTorch 的手写数字识别应用，用户可以在界面上手写数字，应用会实时识别并显示结果。

## 功能特点

- 手写数字绘制画板（基于 OpenCV 窗口）
- 实时数字识别
- 置信度显示
- 清除画板功能
- 轻量级部署方案

## 项目结构

```
handwriting_recognizer_opencv/
├── CMakeLists.txt              # CMake 构建配置文件
├── README.md                   # 项目说明文档
├── main.cpp                    # 应用入口点
├── drawing_canvas.h/cpp        # 手写画板功能
├── torch_model.h/cpp           # LibTorch 模型处理类
└── build/                      # 构建目录
```

## 环境要求

- C++17 或更高版本
- OpenCV 4.x
- LibTorch 1.7.0 或更高版本
- CMake 3.10 或更高版本

## 构建步骤

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
   cmake  -DCMAKE_PREFIX_PATH="D:/dl_torch/libtorch"  -DOpenCV_DIR="D:/opencv/build"
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
./handwriting_recognizer_opencv
```

或者指定模型文件路径：

```bash
./handwriting_recognizer_opencv /path/to/mnist_resnet18_traced.pt
```

## 使用说明

1. 在窗口中用鼠标左键绘制数字
2. 按 'r' 键进行数字识别
3. 查看控制台输出的识别结果和置信度
4. 按 'c' 键清空画板
5. 按 ESC 键退出应用

## 界面操作

- **鼠标左键拖拽**: 在画板上绘制数字
- **'r' 或 'R' 键**: 识别当前绘制的数字
- **'c' 或 'C' 键**: 清除画板
- **ESC 键**: 退出应用

## 部署方案特点

- 轻量级: 不依赖 Python 环境
- 高性能: 利用 LibTorch 优化的推理引擎
- 跨平台: 可在 Windows、Linux 和 macOS 上运行
- 易部署: 只需可执行文件、模型文件和必要的动态库

## 故障排除

如果在构建过程中遇到问题，请参考 [BUILD_ISSUES.md](BUILD_ISSUES.md) 文件，其中包含了常见问题的解决方案。

## 最近修复

本项目最近修复了以下问题：

1. **张量维度不匹配问题**：修正了模型推理时的张量维度处理，确保输入张量形状正确
2. **界面显示问题**：优化了界面文字显示，增大了字体大小和画布尺寸，提高用户体验
3. **编码问题**：将所有中文注释翻译为英文，减少编码相关的警告
