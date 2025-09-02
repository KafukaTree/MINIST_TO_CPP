#include "mainwindow.h"
#include <QApplication>
#include <QDir>
#include <QDebug>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    // 设置应用程序属性
    app.setApplicationName("手写数字识别");
    app.setApplicationVersion("1.0");
    
    // 创建主窗口
    MainWindow window;
    
    // 显示窗口
    window.show();
    
    // 运行应用程序
    return app.exec();
}