#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

// 图像预处理函数
cv::Mat preprocess_image(const cv::Mat& input_image) {
    // 调整图像大小为28x28像素（MNIST标准尺寸）
    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
    
    // 转换为浮点型并归一化到0-1范围
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    
    // 标准化（使用MNIST的均值和标准差）
    float mean = 0.1307;
    float std = 0.3081;
    cv::Mat normalized_image = (float_image - mean) / std;
    
    return normalized_image;
}

// 将OpenCV图像转换为PyTorch张量
torch::Tensor image_to_tensor(const cv::Mat& image) {
    // 创建张量数据
    torch::Tensor tensor = torch::from_blob(image.data, {1, 28, 28}, torch::kFloat32);
    
    // 添加批次维度
    tensor = tensor.unsqueeze(0);
    
    return tensor;
}

int main(int argc, const char* argv[]) {
    // 检查命令行参数
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    
    // 获取模型路径和图像路径
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    
    try {
        // 加载模型
        std::cout << "Loading model from " << model_path << std::endl;
        torch::jit::script::Module module = torch::jit::load(model_path);
        std::cout << "Model loaded successfully!" << std::endl;
        
        // 设置模型为评估模式
        module.eval();
        
        // 加载图像
        std::cout << "Loading image from " << image_path << std::endl;
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Error: Could not load image from " << image_path << std::endl;
            return -1;
        }
        
        // 预处理图像
        std::cout << "Preprocessing image..." << std::endl;
        cv::Mat processed_image = preprocess_image(image);
        
        // 将图像转换为张量
        torch::Tensor input_tensor = image_to_tensor(processed_image);
        
        // 进行推理
        std::cout << "Running inference..." << std::endl;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // 执行前向传播
        at::Tensor output = module.forward(inputs).toTensor();
        
        // 获取预测结果
        auto result = torch::softmax(output, 1);
        auto prediction = result.argmax(1);
        
        // 获取置信度
        auto confidence = result.max(1);
        
        // 输出结果
        std::cout << "Predicted digit: " << prediction[0].item<int>() << std::endl;
        std::cout << "Confidence: " << confidence.values()[0].item<float>() << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "Error: " << e.msg() << std::endl;
        return -1;
    }
    
    return 0;
}