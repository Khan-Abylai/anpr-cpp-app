//
// Created by kartykbayev on 4/13/23.
//

#include "CarDimensions.h"

CarDimensions::CarDimensions(int centerX, int centerY, int width, int height, int frameWidth, int frameHeight) {
    center = cv::Point(centerX, centerY);

    width += extendPixels;
    height += extendPixels;

    this->width = width;
    this->height = height;

    groundCenter = cv::Point(centerX, centerY + height / 2);

    int x1 = centerX - width / 2.0;
    int y1 = centerY - height / 2.0;

    x1 = x1 > 0 ? x1 : 0;
    y1 = y1 > 0 ? y1 : 0;

    topLeft = cv::Point(x1, y1);

    int x2 = centerX + width / 2.0;
    int y2 = centerY + height / 2.0;

    x2 = x2 < frameWidth ? x2 : frameWidth - 1;
    y2 = y2 < frameHeight ? y2 : frameHeight - 1;

    bottomRight = cv::Point(x2, y2);
}

float CarDimensions::getWidth() const {
    return (float) width;
}

float CarDimensions::getHeight() const {
    return (float) height;
}

float CarDimensions::getArea() const {
    return (float) height * (float) width;
}

const cv::Point &CarDimensions::getCenter() const {
    return center;
}

const cv::Point &CarDimensions::getTopLeft() const {
    return topLeft;
}

const cv::Point &CarDimensions::getBottomRight() const {
    return bottomRight;
}
