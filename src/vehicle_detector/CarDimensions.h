//
// Created by kartykbayev on 4/13/23.
//
#pragma once

#include <opencv2/opencv.hpp>

class CarDimensions {
public:
    CarDimensions(int centerX, int centerY, int width, int height, int frameWidth, int frameHeight);

    CarDimensions() = default;

    float getWidth() const;

    float getHeight() const;

    float getArea() const;

    const cv::Point &getCenter() const;

    const cv::Point &getTopLeft() const;

    const cv::Point &getBottomRight() const;
private:
    cv::Point center;
    cv::Point2f groundCenter;
    int width = 0, height = 0;

    cv::Point topLeft, bottomRight;

    int extendPixels = 50;
};
