//
// Created by kartykbayev on 11/2/23.
//
#pragma once

#include <memory>
#include <vector>
#include <cmath>
#include <fstream>
#include <cassert>

#include "NvInfer.h"
#include "../app/TensorRTDeleter.h"
#include "../app/TrtLogger.h"
#include "../app/TensorRTEngine.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <cmath>
#include "../app/Constants.h"


class MobileNetV2 {
public:
    explicit MobileNetV2(Constants::VehicleClassificationType type);

    nvinfer1::IExecutionContext *createExecutionContext();

private:
    void createEngine();

    nvinfer1::ITensor *batchNorm(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network,
                                 int &index,
                                 std::vector<float> &weights,
                                 int channels);

    nvinfer1::ITensor *convBnRelu(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network,
                                  int &index, std::vector<float> &weights, int outChannels,
                                  int kernelSize, int strideStep, int numGroups);

    nvinfer1::ITensor *invertedResidual(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network,
                                        int &index, std::vector<float> &weights,
                                        int inChannels, int outChannels, int expandRatio, int strideStep);

    nvinfer1::ICudaEngine *engine = nullptr;

    const int
            MAX_BATCH_SIZE = 1,
            IMG_WIDTH = 224,
            IMG_HEIGHT = 224,
            IMG_CHANNELS = 3,
            OUTPUT_SIZE = 5;

    std::string ENGINE_NAME, WEIGHTS_FILENAME;

    const std::string NETWORK_INPUT_NAME = "INPUT";

};
