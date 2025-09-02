#ifndef TORCH_MODEL_H
#define TORCH_MODEL_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

class TorchModel {
public:
    TorchModel();
    ~TorchModel();

    // Load model
    bool loadModel(const std::string& modelPath);
    
    // Perform inference
    int predict(const cv::Mat& image, float& confidence);

private:
    // Image preprocessing
    torch::Tensor preprocessImage(const cv::Mat& image);
    
    // Model instance
    std::shared_ptr<torch::jit::script::Module> module;
};

#endif // TORCH_MODEL_H