//
// Created by kartykbayev on 4/13/23.
//
#pragma once

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CarDimensions.h"
#include "../app/TensorRTDeleter.h"

class CarDetection {

public:
    CarDetection(nvinfer1::IExecutionContext *executionContext);

    std::vector<std::shared_ptr<CarDimensions>> carDetect(const cv::Mat &frame);

    ~CarDetection();

private:

    std::vector<float> executeEngine(const cv::Mat &frame);

    std::vector<float> prepareImage(const cv::Mat &frame);

    std::vector<std::shared_ptr<CarDimensions>>
    nms(const std::vector<std::tuple<float, std::shared_ptr<CarDimensions>>> &carDimensions);

    float iou(const std::shared_ptr<CarDimensions> &firstCarDimension,
              const std::shared_ptr<CarDimensions> &secondCarDimension);

    std::vector<std::shared_ptr<CarDimensions>>
    getCarDimensions(std::vector<float> carPredictions, int frameWidth, int frameHeight);
    float PIXEL_MAX_VALUE = 255.0;
    const int
            MAX_BATCH_SIZE = 1,
            CAR_GRID_SIZE = 64,
            CAR_COORDINATE_SIZE = 5,
            IMG_WIDTH = 640,
            IMG_HEIGHT = 640,
            IMG_CHANNELS = 3,

            CAR_GRID_WIDTH = IMG_WIDTH / CAR_GRID_SIZE,
            CAR_GRID_HEIGHT = IMG_HEIGHT / CAR_GRID_SIZE,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            CAR_OUTPUT_SIZE = CAR_COORDINATE_SIZE * CAR_GRID_HEIGHT * CAR_GRID_WIDTH;

    const float CAR_NMS_THRESHOLD = 0.4;
    const float CAR_PROB_THRESHOLD = 0.65;

    nvinfer1::IExecutionContext *executionContext;

    void *cudaBuffer[2];
    cudaStream_t stream;
};