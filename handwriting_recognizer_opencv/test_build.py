#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试构建脚本
此脚本用于验证修复后的代码是否能正常编译
"""

import os
import sys
import subprocess
import shutil

def check_dependencies():
    """检查必要的依赖是否已安装"""
    print("检查依赖项...")
    
    # 检查CMake是否已安装
    try:
        result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] CMake 已安装")
            print(result.stdout.strip())
        else:
            print("[Error] CMake 未安装或不可用")
            return False
    except FileNotFoundError:
        print("[Error] CMake 未安装")
        return False
    
    return True

def check_files():
    """检查必要的文件是否存在"""
    required_files = [
        "handwriting_recognizer_opencv/CMakeLists.txt",
        "handwriting_recognizer_opencv/main.cpp",
        "handwriting_recognizer_opencv/drawing_canvas.h",
        "handwriting_recognizer_opencv/drawing_canvas.cpp",
        "handwriting_recognizer_opencv/torch_model.h",
        "handwriting_recognizer_opencv/torch_model.cpp",
        "mnist_resnet18_traced.pt"
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

def create_build_directory():
    """创建构建目录"""
    if not os.path.exists("build"):
        os.makedirs("build")
        print("\n[OK] 创建构建目录")
    else:
        print("\n[OK] 构建目录已存在")

def configure_cmake():
    """配置CMake"""
    print("\n配置CMake...")
    try:
        # 这里只是一个示例命令，实际路径需要根据用户的环境调整
        result = subprocess.run([
            "cmake", 
            "-DCMAKE_BUILD_TYPE=Release",
            ".."
        ], cwd="build", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] CMake 配置成功")
            return True
        else:
            print("[Error] CMake 配置失败")
            print("错误信息:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[Error] CMake 配置异常: {e}")
        return False

def main():
    print("手写数字识别应用构建测试脚本")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装必要的依赖项。")
        return
    
    # 检查文件
    if not check_files():
        print("\n请确保所有必要的文件都存在。")
        return
    
    # 创建构建目录
    create_build_directory()
    
    # 配置CMake
    # 注意：实际构建需要用户根据自己的环境配置LibTorch路径
    print("\n注意：完整构建需要正确配置LibTorch和OpenCV路径。")
    print("请参考 BUILD_ISSUES.md 文件了解详细配置说明。")
    
    print("\n" + "=" * 50)
    print("构建测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()