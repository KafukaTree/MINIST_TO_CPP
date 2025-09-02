import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
from PIL import Image, ImageDraw
from model import get_model

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别器 (ResNet18)")
        self.root.geometry("400x500") # 设置窗口大小
        self.root.resizable(False, False)
        
        # 创建绘图画布
        self.canvas_width = 300
        self.canvas_height = 300
        
        # 创建一个PIL图像用于绘图
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # 创建GUI组件
        self.create_widgets()
        
        # 加载训练好的模型
        self.load_model()
        
        # 绘图变量
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题黑体左对齐
        title_label = ttk.Label(main_frame, text="  手写数字识别器", font=("SimHei", 16, "bold")) 
        title_label.grid(row=0, column=0, columnspan=1, pady=(0, 10)) # pady=(0, 10) 添加了上下边距
        
        # Logo
        logo_label = ttk.Label(main_frame, text="prod.by.KaFuKa_Tree", font=("Arial", 10, "italic"), foreground="gray") # foreground="gray" 设置文本颜色为灰色
        logo_label.grid(row=0, column=1, sticky=tk.E, pady=(0, 10))
        
        # 绘图画布
        self.canvas = tk.Canvas(
            main_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="black", 
            cursor="cross"
        )
        self.canvas.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        # 识别按钮
        self.recognize_button = ttk.Button(
            button_frame, 
            text="识别数字", 
            command=self.recognize_digit,
            state=tk.DISABLED
        )
        self.recognize_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 清除按钮
        self.clear_button = ttk.Button(
            button_frame, 
            text="清除画布", 
            command=self.clear_canvas
        )
        self.clear_button.pack(side=tk.LEFT)
        
        # 结果框架
        result_frame = ttk.LabelFrame(main_frame, text="识别结果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 结果标签
        self.result_label = ttk.Label(
            result_frame, 
            text="请在上方黑色区域绘制一个数字", 
            font=("Arial", 12),
            foreground="blue"
        )
        self.result_label.pack()
        
        # 置信度标签
        self.confidence_label = ttk.Label(
            result_frame, 
            text="", 
            font=("Arial", 10)
        )
        self.confidence_label.pack()
        
        # 说明标签
        instruction_label = ttk.Label(
            main_frame, 
            text="使用鼠标在黑色区域绘制数字，然后点击'识别数字'按钮",
            font=("Arial", 10),
            foreground="gray"
        )
        instruction_label.grid(row=4, column=0, columnspan=2, pady=(10, 0))
        
    def start_draw(self, event):
        self.old_x = event.x
        self.old_y = event.y
        self.recognize_button.config(state=tk.NORMAL)
        
    def draw_line(self, event):
        if self.old_x and self.old_y:
            # 在Canvas上绘制
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=self.line_width, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE
            )
            
            # 在PIL图像上绘制
            self.draw.line(
                [self.old_x, self.old_y, event.x, event.y],
                fill="white", width=self.line_width
            )
            
        self.old_x = event.x
        self.old_y = event.y
        
    def reset_coords(self, event):
        self.old_x = None
        self.old_y = None
        
    def clear_canvas(self):
        # 清除Canvas
        self.canvas.delete("all")
        
        # 清除PIL图像
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # 重置按钮状态和结果标签
        self.recognize_button.config(state=tk.DISABLED)
        self.result_label.config(text="请在上方黑色区域绘制一个数字")
        self.confidence_label.config(text="")
        
    def load_model(self):
        try:
            # 创建模型实例
            self.model = get_model(num_classes=10)
            
            # 加载模型权重
            self.model.load_state_dict(torch.load("mnist_resnet18.pth", map_location=torch.device('cpu')))
            
            # 设置为评估模式
            self.model.eval()
            
            self.result_label.config(text="模型加载成功！请绘制一个数字进行识别", foreground="green")
        except Exception as e:
            self.result_label.config(text=f"模型加载失败: {str(e)}", foreground="red")
            self.recognize_button.config(state=tk.DISABLED)
            
    def recognize_digit(self):
        try:
            # 预处理图像
            processed_image = self.preprocess_image()
            
            # 使用模型进行预测
            with torch.no_grad():
                output = self.model(processed_image)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            # 显示结果
            digit = predicted.item()
            conf = confidence.item()
            
            self.result_label.config(
                text=f"识别结果: {digit}", 
                font=("Arial", 14, "bold"),
                foreground="blue"
            )
            self.confidence_label.config(
                text=f"置信度: {conf:.2%}"
            )
            
        except Exception as e:
            self.result_label.config(text=f"识别失败: {str(e)}", foreground="red")
            self.confidence_label.config(text="")
            
    def preprocess_image(self):
        # 复制图像并进行预处理
        img = self.image.copy()
        
        # 调整大小为28x28像素（MNIST标准尺寸）
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 归一化到0-1范围
        img_array = img_array.astype(np.float32) / 255.0
        
        # 标准化（使用MNIST的均值和标准差）
        mean = 0.1307
        std = 0.3081
        img_array = (img_array - mean) / std
        
        # 添加批次维度和通道维度
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor

def main():
    try:
        root = tk.Tk()
        app = DigitRecognizerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"GUI应用程序错误: {e}")
        return False
    return True

if __name__ == "__main__":
    main()