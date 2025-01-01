//
// Created by artyk on 11/9/2023.
//
#pragma once

#include <filesystem>
#include <memory>
#include <vector>
#include <cmath>
#include <fstream>

#include "NvInfer.h"

#include "../app/TensorRTDeleter.h"
#include "../app/TrtLogger.h"
#include "../app/TensorRTEngine.h"
#include "../app/Constants.h"


class MobileNet {
public:
    MobileNet();

    nvinfer1::IExecutionContext *createExecutionContext();

private:
    void createEngine();

    nvinfer1::ITensor *batchNorm(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network,
                                 int &index,
                                 std::vector<float> &weights,
                                 int channels);

    nvinfer1::ITensor *convBnRelu(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network,
                                  int &index, std::vector<float> &weights, int inChannels, int outChannels,
                                  int kernelSize, int strideStep, int numGroups);

    nvinfer1::ITensor *invertedResidual(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network,
                                        int &index, std::vector<float> &weights,
                                        int inChannels, int outChannels, int expandRatio, int strideStep);


    nvinfer1::ICudaEngine *engine = nullptr;

    const int
            MAX_BATCH_SIZE = 1,
            IMG_WIDTH = 320,
            IMG_HEIGHT = 320,
            IMG_CHANNELS = Constants::IMG_CHANNELS,
            CAR_MODEL_OUTPUT_SIZE = 417,
            CAR_COLOR_OUTPUT_SIZE = 4,
            OUTPUT_SIZE = CAR_MODEL_OUTPUT_SIZE + CAR_COLOR_OUTPUT_SIZE;

    const std::string NETWORK_INPUT_NAME = "INPUT",
            ENGINE_NAME = "mobilenet.engine",
            WEIGHTS_FILENAME = "mobilenet.np";

};


