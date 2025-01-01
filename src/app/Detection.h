#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "LicensePlate.h"
#include "Constants.h"
#include "TensorRTDeleter.h"
#include "../ILogger.h"

class Detection : public ::ILogger {
public:

    Detection(nvinfer1::IExecutionContext *executionContext, const float &detectionThreshold);

    std::vector<std::shared_ptr<LicensePlate>> detect(const cv::Mat &frame);

    ~Detection();

private:
    const int
            MAX_BATCH_SIZE = Constants::DETECTION_BATCH_SIZE,
            PLATE_COORDINATE_SIZE = Constants::PLATE_COORDINATE_SIZE,
            IMG_WIDTH = Constants::DETECTION_IMG_W,
            IMG_HEIGHT = Constants::DETECTION_IMG_H,
            IMG_CHANNELS = Constants::IMG_CHANNELS,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            PLATE_GRID_SIZE = 16,
            PLATE_GRID_WIDTH = IMG_WIDTH / PLATE_GRID_SIZE,
            PLATE_GRID_HEIGHT = IMG_HEIGHT / PLATE_GRID_SIZE,
            PLATE_OUTPUT_SIZE = PLATE_COORDINATE_SIZE * PLATE_GRID_HEIGHT * PLATE_GRID_WIDTH;

    const std::vector<int> PLATE_GRID_SIZES{8, 16, 32, 64};

    const float LP_NMS_THRESHOLD = 0.4;
    const float LP_PROB_THRESHOLD = 0.8;

    nvinfer1::IExecutionContext *executionContext;

    void *cudaBuffer[2]{};
    cudaStream_t stream{};

    std::vector<float> executeEngine(const cv::Mat &frame);

    [[nodiscard]] std::vector<float> prepareImage(const cv::Mat &frame) const;

    [[nodiscard]] std::vector<std::shared_ptr<LicensePlate>>
    nms(const std::vector<std::tuple<float, std::shared_ptr<LicensePlate>>> &licensePlates) const;

    static float iou(const std::shared_ptr<LicensePlate> &firstLp, const std::shared_ptr<LicensePlate> &secondLp);

    [[nodiscard]] std::vector<std::tuple<float, std::shared_ptr<LicensePlate>>>
    getLicensePlates(std::vector<float> lpPredictions, int frameWidth, int frameHeight) const;

};