import cv2
import numpy as np

def create_test_image():
    # 创建一个黑色背景的图像 (28x28 pixels)
    image = np.zeros((28, 28), dtype=np.uint8)
    
    # 绘制数字 "8" 的简单表示
    # 绘制两个圆圈来表示数字8
    cv2.circle(image, (14, 9), 4, 255, -1)   # 上面的圆圈
    cv2.circle(image, (14, 19), 4, 255, -1)  # 下面的圆圈
    
    # 保存图像
    cv2.imwrite("test_image.png", image)
    print("测试图像已保存为 test_image.png")
    
    # 显示图像 (可选)
    cv2.imshow("Test Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_test_image()