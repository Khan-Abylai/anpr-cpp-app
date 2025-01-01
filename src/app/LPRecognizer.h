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


    std::tuple<std::string, double> makePrediction(const std::vector<cv::Mat> &frames);

private:

    const int
            SEQUENCE_SIZE = Constants::EU_SEQUENCE_SIZE,
            ALPHABET_SIZE = 38,
            BLANK_INDEX = 0,
            IMG_WIDTH = Constants::EU_LP_W,
            IMG_HEIGHT = Constants::EU_LP_H,
            IMG_CHANNELS = Constants::IMG_CHANNELS,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            OUTPUT_SIZE = SEQUENCE_SIZE * ALPHABET_SIZE,
            // REGION_SIZE = 37,
            // OUTPUT_2_SIZE = REGION_SIZE,
            MAX_BATCH_SIZE = Constants::RECOGNIZER_MAX_BATCH_SIZE,
            MAX_PLATE_SIZE = 12;

    const std::string
            ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz",
            NETWORK_INPUT_NAME = "INPUT",
            NETWORK_DIM_NAME = "DIMENSIONS",
            NETWORK_OUTPUT_NAME = "OUTPUT",
            // NETWORK_OUTPUT_2_NAME = "OUTPUT_2",
            ENGINE_NAME = "recognizer_malaysia.engine",
            WEIGHTS_FILENAME = "recognizer_malaysia.np";

    std::vector<int> dimensions;

    void createEngine();

    // std::vector<std::string> REGIONS{"albania", "andorra", "austria", "belgium", "bosnia", "bulgaria", "croatia", "cyprus", "czech", "estonia",
    //                                  "finland", "france", "germany", "greece", "hungary", "ireland", "italy", "latvia",
    //                                  "licht", "lithuania", "luxemburg", "makedonia", "malta", "monaco", "montenegro", "netherlands", "poland",
    //                                  "portugal", "romania", "san_marino", "serbia", "slovakia", "slovenia", "spain", "sweden", "swiss", "marocco"};

    std::vector<float> executeInferEngine(const std::vector<cv::Mat> &frames);

    std::vector<float> prepareImage(const std::vector<cv::Mat> &frames);

    std::vector<float> softmax(std::vector<float> &score_vec);

    void *cudaBuffer[3];

    nvinfer1::IExecutionContext *executionContext;
    nvinfer1::ICudaEngine *engine = nullptr;
    cudaStream_t stream;
};