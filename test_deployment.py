#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 LibTorch + C++ 部署方案的脚本
此脚本用于说明如何测试部署方案，而不是实际执行构建和测试
"""

import os
import sys

def print_test_instructions():
    """打印测试说明"""
    print("=" * 60)
    print("MNIST 手写数字识别 LibTorch C++ 部署方案测试说明")
    print("=" * 60)
    
    print("\n1. 环境准备:")
    print("   - 确保已安装 LibTorch (C++ 版本的 PyTorch)")
    print("   - 确保已安装 OpenCV C++ 库")
    print("   - 确保已安装 CMake (3.10 或更高版本)")
    
    print("\n2. 构建项目:")
    print("   mkdir build")
    print("   cd build")
    print("   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..")
    print("   cmake --build . --config Release")
    
    print("\n3. 运行测试:")
    print("   # Linux/macOS:")
    print("   ./mnist_classifier ../mnist_resnet18_traced.pt ../test_image.png")
    print("\n   # Windows:")
    print("   .\\Release\\mnist_classifier.exe ..\\mnist_resnet18_traced.pt ..\\test_image.png")
    
    print("\n4. 预期输出:")
    print("   Loading model from ../mnist_resnet18_traced.pt")
    print("   Model loaded successfully!")
    print("   Loading image from ../test_image.png")
    print("   Preprocessing image...")
    print("   Running inference...")
    print("   Predicted digit: 8")
    print("   Confidence: 0.95")
    
    print("\n5. 部署方案特点:")
    print("   - 轻量级: 不依赖 Python 环境")
    print("   - 高性能: 利用 LibTorch 优化的推理引擎")
    print("   - 跨平台: 可在 Windows、Linux 和 macOS 上运行")
    print("   - 易部署: 只需可执行文件和模型文件")

def check_files():
    """检查必要的文件是否存在"""
    required_files = [
        "mnist_resnet18_traced.pt",
        "test_image.png",
        "CMakeLists.txt",
        "src/main.cpp"
    ]
    
    print("\n检查必要的文件:")
    all_files_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  [OK] {file}")
        else:
            print(f"  [Missing] {file}")
            all_files_exist = False
    
    return all_files_exist

def main():
    print("LibTorch + C++ 部署方案测试脚本")
    
    # 检查文件
    files_ok = check_files()
    
    if files_ok:
        print("\n[OK] 所有必要的文件都已存在，可以进行构建和测试。")
    else:
        print("\n[Error] 缺少必要的文件，请确保已运行模型转换脚本和图像创建脚本。")
    
    # 打印测试说明
    print_test_instructions()
    
    print("\n" + "=" * 60)
    print("测试说明已完成")
    print("=" * 60)

if __name__ == "__main__":
    main()