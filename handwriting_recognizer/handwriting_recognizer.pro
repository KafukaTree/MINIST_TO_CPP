QT += core gui widgets

CONFIG += c++17

TARGET = handwriting_recognizer
TEMPLATE = app

# Find and link LibTorch
LIBS += -ltorch -ltorch_cpu -lc10 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

INCLUDEPATH += /path/to/libtorch/include \
               /path/to/libtorch/include/torch/csrc/api/include \
               /path/to/opencv/include

LIBS += -L/path/to/libtorch/lib \
        -L/path/to/opencv/lib

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    drawingwidget.cpp \
    torchmodel.cpp

HEADERS += \
    mainwindow.h \
    drawingwidget.h \
    torchmodel.h

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target