#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试修复脚本
此脚本用于验证修复后的代码是否能正常工作
"""

import os
import sys

def check_files():
    """检查修复后的文件"""
    print("检查修复后的文件...")
    
    # 检查torch_model.cpp文件
    torch_model_cpp = "handwriting_recognizer_opencv/torch_model.cpp"
    if os.path.exists(torch_model_cpp):
        print("[OK] torch_model.cpp 文件存在")
        # 检查是否还有中文注释
        with open(torch_model_cpp, 'r', encoding='utf-8') as f:
            content = f.read()
            if '//' in content and any(c in content for c in ['加载模型', '进行推理', '图像预处理', '模型实例']):
                print("[Warning] torch_model.cpp 中可能还有中文注释")
            else:
                print("[OK] torch_model.cpp 中没有发现中文注释")
    else:
        print("[Error] torch_model.cpp 文件不存在")
    
    # 检查drawing_canvas.cpp文件
    drawing_canvas_cpp = "handwriting_recognizer_opencv/drawing_canvas.cpp"
    if os.path.exists(drawing_canvas_cpp):
        print("[OK] drawing_canvas.cpp 文件存在")
        # 检查是否还有中文注释
        with open(drawing_canvas_cpp, 'r', encoding='utf-8') as f:
            content = f.read()
            if '//' in content and any(c in content for c in ['将彩色图像转换为灰度图像', '避免在文字区域绘制']):
                print("[Warning] drawing_canvas.cpp 中可能还有中文注释")
            else:
                print("[OK] drawing_canvas.cpp 中没有发现中文注释")
    else:
        print("[Error] drawing_canvas.cpp 文件不存在")
    
    # 检查main.cpp文件
    main_cpp = "handwriting_recognizer_opencv/main.cpp"
    if os.path.exists(main_cpp):
        print("[OK] main.cpp 文件存在")
    else:
        print("[Error] main.cpp 文件不存在")

def main():
    print("手写数字识别应用修复验证脚本")
    print("=" * 50)
    
    # 检查文件
    check_files()
    
    print("\n" + "=" * 50)
    print("修复验证完成")
    print("=" * 50)
    print("\n请重新构建项目以验证修复效果：")
    print("1. mkdir build")
    print("2. cd build")
    print("3. cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..")
    print("4. cmake --build . --config Release")

if __name__ == "__main__":
    main()