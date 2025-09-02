#include "drawingwidget.h"
#include <QPainter>
#include <QMouseEvent>
#include <QResizeEvent>

DrawingWidget::DrawingWidget(QWidget *parent)
    : QWidget(parent)
    , isDrawing(false)
    , penWidth(15)
{
    // 设置背景为黑色
    setAutoFillBackground(true);
    QPalette palette = this->palette();
    palette.setColor(QPalette::Window, Qt::black);
    setPalette(palette);
    
    // 启用鼠标跟踪
    setMouseTracking(true);
}

DrawingWidget::~DrawingWidget()
{
}

QImage DrawingWidget::getImage() const
{
    return image;
}

void DrawingWidget::clear()
{
    image.fill(qRgb(0, 0, 0));  // 填充为黑色
    update();
}

void DrawingWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        lastPoint = event->pos();
        isDrawing = true;
    }
}

void DrawingWidget::mouseMoveEvent(QMouseEvent *event)
{
    if ((event->buttons() & Qt::LeftButton) && isDrawing) {
        drawLineTo(event->pos());
    }
}

void DrawingWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && isDrawing) {
        drawLineTo(event->pos());
        isDrawing = false;
    }
}

void DrawingWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    QRect dirtyRect = event->rect();
    painter.drawImage(dirtyRect, image, dirtyRect);
}

void DrawingWidget::resizeEvent(QResizeEvent *event)
{
    if (image.size() != size()) {
        // 调整图像大小以匹配窗口大小
        QImage newImage(size(), QImage::Format_RGB32);
        newImage.fill(qRgb(0, 0, 0));  // 填充为黑色
        
        // 将原图像绘制到新图像上
        QPainter painter(&newImage);
        painter.drawImage(QPoint(0, 0), image);
        
        image = newImage;
    }
    
    QWidget::resizeEvent(event);
}

void DrawingWidget::drawLineTo(const QPoint &endPoint)
{
    // 确保图像大小正确
    if (image.size() != size()) {
        QImage newImage(size(), QImage::Format_RGB32);
        newImage.fill(qRgb(0, 0, 0));  // 填充为黑色
        image = newImage;
    }
    
    QPainter painter(&image);
    painter.setPen(QPen(Qt::white, penWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(lastPoint, endPoint);
    
    lastPoint = endPoint;
    
    update();
}