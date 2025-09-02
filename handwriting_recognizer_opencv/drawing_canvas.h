#ifndef DRAWING_CANVAS_H
#define DRAWING_CANVAS_H

#include <opencv2/opencv.hpp>
#include <string>

class DrawingCanvas {
public:
    DrawingCanvas(int width = 300, int height = 300);
    ~DrawingCanvas();

    // Show canvas window
    void show(const std::string& windowName = "Handwriting Canvas");
    
    // Run main loop
    bool run();  // Return true to recognize, false to exit
    
    // Get drawn image
    cv::Mat getImage() const;
    
    // Clear canvas
    void clear();
    
    // Set recognition result
    void setResult(const std::string& result);
    
    // Check if recognition is needed
    bool shouldRecognize() const;
    
    // Check if clearing is needed
    bool shouldClear() const;
    
    // Reset state
    void resetState();
    
private:
    // 鼠标回调函数
    static void onMouse(int event, int x, int y, int flags, void* userdata);
    
    // 绘制界面
    void drawInterface();
    
    // 处理鼠标事件
    void handleMouse(int event, int x, int y);
    
    cv::Mat canvas;
    cv::Point lastPoint;
    bool isDrawing;
    int penWidth;
    std::string windowName;
    std::string resultText;
    bool recognizeFlag;
    bool clearFlag;
    bool exitFlag;
};

#endif // DRAWING_CANVAS_H