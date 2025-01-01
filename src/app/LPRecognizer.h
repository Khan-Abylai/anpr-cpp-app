#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <array>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "Constants.h"
#include "TensorRTDeleter.h"
#include "TrtLogger.h"
#include "TensorRTEngine.h"

class LPRecognizer {

public:

    LPRecognizer();

    ~LPRecognizer();

    std::vector<std::pair<std::string, float>> predict(const std::vector<cv::Mat> &frames);

private:

    const int
            SEQUENCE_SIZE = 30,
            ALPHABET_SIZE = 38,
            BLANK_INDEX = 0,
            IMG_WIDTH = Constants::RECT_LP_W,
            IMG_HEIGHT = Constants::RECT_LP_H,
            IMG_CHANNELS = Constants::IMG_CHANNELS,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            OUTPUT_SIZE = SEQUENCE_SIZE * ALPHABET_SIZE,
            MAX_BATCH_SIZE = Constants::RECOGNIZER_MAX_BATCH_SIZE,
            MAX_PLATE_SIZE = 12;

    const std::string
            ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz.",
            NETWORK_INPUT_NAME = "INPUT",
            NETWORK_DIM_NAME = "DIMENSIONS",
            NETWORK_OUTPUT_NAME = "OUTPUT",
            ENGINE_NAME = "recognizer.engine",
            WEIGHTS_FILENAME = "recognizer_main_2024.np";
    std::vector<int> dimensions;

    void createEngine();

    std::vector<float> executeEngine(const std::vector<cv::Mat> &frames);

    [[nodiscard]] std::vector<float> prepareImage(const std::vector<cv::Mat> &frames) const;

    void *cudaBuffer[3]{};

    nvinfer1::IExecutionContext *executionContext;
    nvinfer1::ICudaEngine *engine = nullptr;
    cudaStream_t stream{};
};