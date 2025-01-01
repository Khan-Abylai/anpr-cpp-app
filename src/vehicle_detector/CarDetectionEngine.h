//
// Created by kartykbayev on 4/13/23.
//

#pragma once

#include <vector>
#include <array>
#include <utility>
#include <string>
#include <fstream>
#include <unordered_set>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include "../app/TrtLogger.h"
#include "../app/TensorRTEngine.h"
#include "../app/TensorRTDeleter.h"
#include "../app/Constants.h"
class CarDetectionEngine {
private:
    nvinfer1::ICudaEngine *engine = nullptr;

    const int MAX_BATCH_SIZE = 1, CAR_COORDINATE_SIZE = 5, IMG_WIDTH = 640, IMG_HEIGHT = 640, IMG_CHANNELS = 3;

    const int COORDINATE_SIZES[1] = {CAR_COORDINATE_SIZE};

    const std::string NETWORK_INPUT_NAME = "INPUT", ENGINE_NAME = "car_detection.engine", WEIGHTS_FILENAME = "car_detector_weights_3.np";
    std::string NETWORK_OUTPUT_NAMES[1] = {"CAR_OUTPUT"};

    void createEngine();

public:
    nvinfer1::IExecutionContext *createExecutionContext();

    CarDetectionEngine();
};