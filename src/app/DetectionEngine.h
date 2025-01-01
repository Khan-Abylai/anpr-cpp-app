#pragma once

#include <vector>
#include <array>
#include <utility>
#include <string>
#include <fstream>
#include <filesystem>
#include "unordered_set"
#include "utility"
#include "vector"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "LicensePlate.h"
#include "Constants.h"
#include "TensorRTDeleter.h"
#include "TrtLogger.h"
#include "TensorRTEngine.h"

class DetectionEngine {
private:

    nvinfer1::ICudaEngine *engine = nullptr;

    const int
            MAX_BATCH_SIZE = Constants::DETECTION_BATCH_SIZE,
            COORDINATE_SIZE = Constants::PLATE_COORDINATE_SIZE,
            IMG_WIDTH = Constants::DETECTION_IMG_W,
            IMG_HEIGHT = Constants::DETECTION_IMG_H,
            IMG_CHANNELS = Constants::IMG_CHANNELS;
    const int COORDINATE_SIZES[1] = {COORDINATE_SIZE};
    const std::string NETWORK_INPUT_NAME = "INPUT",
            ENGINE_NAME = "detection.engine",
            WEIGHTS_FILENAME = "detector_marocco_europe.np";
    const std::vector<std::string> NETWORK_OUTPUT_NAMES{"PLATE_OUTPUT"};

    void createEngine();

public:

    nvinfer1::IExecutionContext *createExecutionContext();

    DetectionEngine();

};