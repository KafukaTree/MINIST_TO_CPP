#include "mainwindow.h"
#include "drawingwidget.h"
#include "torchmodel.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>
#include <QFileDialog>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , torchModel(new TorchModel())
{
    setupUI();
    loadModel();
}

MainWindow::~MainWindow()
{
    delete torchModel;
}

void MainWindow::setupUI()
{
    // 创建中心部件
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    // 创建主布局
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    
    // 创建标题
    QLabel *titleLabel = new QLabel("手写数字识别", this);
    titleLabel->setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;");
    titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(titleLabel);
    
    // 创建绘图画板
    drawingWidget = new DrawingWidget(this);
    drawingWidget->setMinimumSize(300, 300);
    mainLayout->addWidget(drawingWidget, 1);
    
    // 创建按钮布局
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    
    recognizeButton = new QPushButton("识别", this);
    clearButton = new QPushButton("清除", this);
    
    buttonLayout->addWidget(recognizeButton);
    buttonLayout->addWidget(clearButton);
    buttonLayout->addStretch();
    
    mainLayout->addLayout(buttonLayout);
    
    // 创建结果显示区域
    QGroupBox *resultGroup = new QGroupBox("识别结果", this);
    QVBoxLayout *resultLayout = new QVBoxLayout(resultGroup);
    
    resultLabel = new QLabel("请在上方区域手写数字", this);
    resultLabel->setStyleSheet("font-size: 24px; font-weight: bold;");
    resultLabel->setAlignment(Qt::AlignCenter);
    
    confidenceLabel = new QLabel("置信度: -", this);
    confidenceLabel->setStyleSheet("font-size: 16px;");
    confidenceLabel->setAlignment(Qt::AlignCenter);
    
    resultLayout->addWidget(resultLabel);
    resultLayout->addWidget(confidenceLabel);
    
    mainLayout->addWidget(resultGroup);
    
    // 连接信号和槽
    connect(recognizeButton, &QPushButton::clicked, this, &MainWindow::onRecognizeClicked);
    connect(clearButton, &QPushButton::clicked, this, &MainWindow::onClearClicked);
    
    // 设置窗口属性
    setWindowTitle("手写数字识别");
    resize(400, 600);
}

void MainWindow::loadModel()
{
    try {
        // 获取模型文件路径
        QString modelPath = QDir::currentPath() + "/../mnist_resnet18_traced.pt";
        
        // 检查模型文件是否存在
        if (!QFile::exists(modelPath)) {
            // 如果在相对路径找不到，尝试在当前目录查找
            modelPath = "mnist_resnet18_traced.pt";
        }
        
        if (!QFile::exists(modelPath)) {
            QMessageBox::warning(this, "警告", "未找到模型文件 mnist_resnet18_traced.pt");
            return;
        }
        
        // 加载模型
        if (torchModel->loadModel(modelPath)) {
            qDebug() << "模型加载成功";
        } else {
            QMessageBox::critical(this, "错误", "模型加载失败");
        }
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", "模型加载异常: " + QString(e.what()));
    }
}

void MainWindow::onRecognizeClicked()
{
    // 获取绘图画板上的图像
    QImage image = drawingWidget->getImage();
    
    if (image.isNull()) {
        QMessageBox::warning(this, "警告", "请先绘制数字");
        return;
    }
    
    // 使用模型进行预测
    float confidence = 0.0;
    int predictedDigit = torchModel->predict(image, confidence);
    
    if (predictedDigit >= 0) {
        // 显示结果
        resultLabel->setText(QString("识别结果: %1").arg(predictedDigit));
        confidenceLabel->setText(QString("置信度: %1%").arg(confidence * 100, 0, 'f', 1));
    } else {
        QMessageBox::warning(this, "警告", "识别失败，请重试");
        resultLabel->setText("识别失败");
        confidenceLabel->setText("置信度: -");
    }
}

void MainWindow::onClearClicked()
{
    // 清除绘图画板
    drawingWidget->clear();
    
    // 重置结果显示
    resultLabel->setText("请在上方区域手写数字");
    confidenceLabel->setText("置信度: -");
}