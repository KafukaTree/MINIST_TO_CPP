#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QGroupBox>

class DrawingWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onRecognizeClicked();
    void onClearClicked();

private:
    void setupUI();
    void loadModel();

    // UI components
    DrawingWidget *drawingWidget;
    QPushButton *recognizeButton;
    QPushButton *clearButton;
    QLabel *resultLabel;
    QLabel *confidenceLabel;
    
    // Model
    void* torchModel;  // 简化的模型指针表示
};

#endif // MAINWINDOW_H