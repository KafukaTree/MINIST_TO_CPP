#include "torchmodel.h"
#include <QImage>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

TorchModel::TorchModel()
{
}

TorchModel::~TorchModel()
{
}

bool TorchModel::loadModel(const QString &modelPath)
{
    try {
        // 加载模型
        module = std::make_shared<torch::jit::script::Module>(torch::jit::load(modelPath.toStdString()));
        
        // 设置模型为评估模式
        module->eval();
        
        return true;
    } catch (const std::exception& e) {
        // 错误处理
        return false;
    }
}

torch::Tensor TorchModel::preprocessImage(const QImage &inputImage)
{
    // 将QImage转换为OpenCV格式
    QImage grayImage = inputImage.convertToFormat(QImage::Format_Grayscale8);
    cv::Mat cvImage(grayImage.height(), grayImage.width(), CV_8UC1, 
                    const_cast<uchar*>(grayImage.bits()), grayImage.bytesPerLine());
    
    // 调整图像大小为28x28像素（MNIST标准尺寸）
    cv::Mat resizedImage;
    cv::resize(cvImage, resizedImage, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
    
    // 转换为浮点型并归一化到0-1范围
    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);
    
    // 标准化（使用MNIST的均值和标准差）
    float mean = 0.1307;
    float std = 0.3081;
    cv::Mat normalizedImage = (floatImage - mean) / std;
    
    // 转换为PyTorch张量
    torch::Tensor tensor = torch::from_blob(normalizedImage.data, {1, 28, 28}, torch::kFloat32);
    
    // 添加批次维度和通道维度
    tensor = tensor.unsqueeze(0).unsqueeze(0);
    
    return tensor;
}

int TorchModel::predict(const QImage &image, float &confidence)
{
    try {
        // 预处理图像
        torch::Tensor inputTensor = preprocessImage(image);
        
        // 进行推理
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        
        // 执行前向传播
        at::Tensor output = module->forward(inputs).toTensor();
        
        // 获取预测结果
        auto result = torch::softmax(output, 1);
        auto prediction = result.argmax(1);
        
        // 获取置信度
        auto confidenceTensor = result.max(1);
        
        // 返回结果
        confidence = confidenceTensor.values()[0].item<float>();
        return prediction[0].item<int>();
    } catch (const std::exception& e) {
        // 错误处理
        confidence = 0.0;
        return -1;  // 表示错误
    }
}