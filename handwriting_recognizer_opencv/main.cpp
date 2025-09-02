#include "drawing_canvas.h"
#include "torch_model.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    std::cout << "Handwriting Digit Recognizer" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Create model instance
    TorchModel model;
    
    // Load model
    std::string modelPath = "../mnist_resnet18_traced.pt";  // Default model path
    if (argc > 1) {
        modelPath = argv[1];  // Use model path specified by command line argument
    }
    
    std::cout << "Loading model from: " << modelPath << std::endl;
    if (!model.loadModel(modelPath)) {
        std::cerr << "Failed to load model. Exiting." << std::endl;
        return -1;
    }
    
    // Create drawing canvas
    DrawingCanvas canvas(500, 500); // Create canvas with width and height of 700 pixels
    canvas.show("Handwriting Digit Recognizer");
    
    // Main loop
    while (canvas.run()) {
        // Check if clearing is needed
        if (canvas.shouldClear()) {
            canvas.clear();
            canvas.show("Handwriting Digit Recognizer");
            continue;
        }
        
        // Get image and perform recognition
        cv::Mat image = canvas.getImage();
        
        // Perform prediction
        float confidence = 0.0;
        int predictedDigit = model.predict(image, confidence);
        
        if (predictedDigit >= 0) {
            std::string result = "Predicted: " + std::to_string(predictedDigit) + 
                                " (Confidence: " + std::to_string(confidence * 100).substr(0, 5) + "%)"; // Truncate confidence to 2 decimal places
            std::cout << result << std::endl;
            canvas.setResult(result);
        } else {
            std::string result = "Recognition failed!";
            std::cout << result << std::endl;
            canvas.setResult(result);
        }
        
        // Redisplay canvas with result
        canvas.show("Handwriting Digit Recognizer"); // Display canvas with updated result
        
        // Clear canvas for next drawing
        canvas.clear();
    }
    
    std::cout << "Goodbye!" << std::endl;
    return 0;
}