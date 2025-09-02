#include "drawing_canvas.h"
#include <opencv2/opencv.hpp>
#include <iostream>

DrawingCanvas::DrawingCanvas(int width, int height)
    : canvas(height, width, CV_8UC3, cv::Scalar(0, 0, 0))
    , isDrawing(false)
    , penWidth(35) // Increased pen width for better visibility
    , recognizeFlag(false)
    , clearFlag(false)
    , exitFlag(false)
{
    // Initialize canvas
    canvas = cv::Scalar(0, 0, 0);  // Black background
}

DrawingCanvas::~DrawingCanvas()
{
    cv::destroyAllWindows();
}

void DrawingCanvas::show(const std::string& windowName)
{
    this->windowName = windowName;
    cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(this->windowName, onMouse, this);
    drawInterface();
    cv::imshow(this->windowName, canvas);
}

bool DrawingCanvas::run()
{
    recognizeFlag = false;
    clearFlag = false;
    exitFlag = false;
    
    while (true) {
        drawInterface();
        cv::imshow(windowName, canvas);
        
        int key = cv::waitKey(1) & 0xFF;
        
        if (key == 27) {  // ESC key to exit
            exitFlag = true;
            break;
        } else if (key == 'r' || key == 'R') {  // R key to recognize
            recognizeFlag = true;
            resultText = "Recognizing...";  // Clear previous result and show recognizing message
            break;
        } else if (key == 'c' || key == 'C') {  // C key to clear
            clearFlag = true;
            break;
        }
    }
    
    return !exitFlag;
}

cv::Mat DrawingCanvas::getImage() const
{
    // Convert color image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(canvas, grayImage, cv::COLOR_BGR2GRAY);
    return grayImage;
}

void DrawingCanvas::clear()
{
    canvas = cv::Scalar(0, 0, 0);  // Reset to black background
}

void DrawingCanvas::setResult(const std::string& result)
{
    resultText = result;
}

bool DrawingCanvas::shouldRecognize() const
{
    return recognizeFlag;
}

bool DrawingCanvas::shouldClear() const
{
    return clearFlag;
}

void DrawingCanvas::resetState()
{
    recognizeFlag = false;
    clearFlag = false;
    exitFlag = false;
}

void DrawingCanvas::onMouse(int event, int x, int y, int flags, void* userdata)
{
    DrawingCanvas* canvas = static_cast<DrawingCanvas*>(userdata);
    if (canvas) {
        canvas->handleMouse(event, x, y);
    }
}

void DrawingCanvas::drawInterface()
{
    // Create a copy for display
    cv::Mat displayCanvas = canvas.clone();
    
    // Display instructions at the top with better visibility
    cv::putText(displayCanvas, "Draw a digit with mouse",
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    
    cv::putText(displayCanvas, "Press 'r' to recognize, 'c' to clear, ESC to exit",
                cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 1);
    
    // Display result with better positioning and size
    cv::putText(displayCanvas, resultText,
                cv::Point(10, canvas.rows - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 255), 2);
    
    // Update display canvas
    canvas = displayCanvas;
}

void DrawingCanvas::handleMouse(int event, int x, int y)
{
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            lastPoint = cv::Point(x, y);
            isDrawing = true;
            break;
            
        case cv::EVENT_MOUSEMOVE:
            if (isDrawing && y > 50) {  // Avoid drawing in text area
                cv::line(canvas, lastPoint, cv::Point(x, y),
                         cv::Scalar(255, 255, 255), penWidth, cv::LINE_AA);
                lastPoint = cv::Point(x, y);
            }
            break;
            
        case cv::EVENT_LBUTTONUP:
            if (isDrawing) {
                isDrawing = false;
            }
            break;
    }
}