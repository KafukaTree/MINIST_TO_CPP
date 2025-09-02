#include "torch_model.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

TorchModel::TorchModel()
{
}

TorchModel::~TorchModel()
{
}

bool TorchModel::loadModel(const std::string& modelPath)
{
    try {
        // Load model
        module = std::make_shared<torch::jit::script::Module>(torch::jit::load(modelPath));
        
        // Set model to evaluation mode
        module->eval();
        
        std::cout << "Model loaded successfully!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

torch::Tensor TorchModel::preprocessImage(const cv::Mat& inputImage)
{
    // Ensure the image is grayscale
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage;
    }
    
    // Resize image to 28x28 pixels (MNIST standard size)
    cv::Mat resizedImage;
    cv::resize(grayImage, resizedImage, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
    
    // Convert to float and normalize to 0-1 range
    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    
    // Normalize (using MNIST mean and standard deviation)
    float mean = 0.1307f;
    float std = 0.3081f;
    cv::Mat normalizedImage = (floatImage - mean) / std;
    
    // Convert to PyTorch tensor (expected shape: [batch, channel, height, width])
    torch::Tensor tensor = torch::from_blob(normalizedImage.data, {28, 28}, torch::kFloat32);
    
    // Add channel dimension and batch dimension
    tensor = tensor.unsqueeze(0).unsqueeze(0);
    
    return tensor;
}

int TorchModel::predict(const cv::Mat& image, float& confidence)
{
    try {
        // Preprocess image
        torch::Tensor inputTensor = preprocessImage(image);
        
        // Perform inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        
        // 执行前向传播
        at::Tensor output = module->forward(inputs).toTensor();
        
        // 获取预测结果
        auto result = torch::softmax(output, 1);
        auto prediction = result.argmax(1);
        
        // 获取置信度
        auto maxResult = result.max(1);
        auto maxValues = std::get<0>(maxResult);  // 最大值
        auto maxIndices = std::get<1>(maxResult);  // 最大值的索引
        
        // 返回结果
        confidence = maxValues[0].item<float>();
        return maxIndices[0].item<int>();
    } catch (const std::exception& e) {
        std::cerr << "Error during prediction: " << e.what() << std::endl;
        confidence = 0.0;
        return -1;  // 表示错误
    }
}