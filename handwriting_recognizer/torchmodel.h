#ifndef TORCHMODEL_H
#define TORCHMODEL_H

#include <QString>
#include <QImage>
#include <torch/torch.h>
#include <torch/script.h>

class TorchModel
{
public:
    TorchModel();
    ~TorchModel();

    // 加载模型
    bool loadModel(const QString &modelPath);
    
    // 进行推理
    int predict(const QImage &image, float &confidence);

private:
    // 图像预处理
    torch::Tensor preprocessImage(const QImage &image);
    
    // 模型实例
    std::shared_ptr<torch::jit::script::Module> module;
};

#endif // TORCHMODEL_H