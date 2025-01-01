//
// Created by kartykbayev on 6/6/22.
//
#pragma once

#include <filesystem>
#include <fstream>
#include <vector>
#include "TensorRTDeleter.h"
#include "NvInfer.h"
#include "TrtLogger.h"

class TensorRTEngine {
public:
    static void serializeEngine(nvinfer1::ICudaEngine *engine, const std::string &engineFilename);

    static nvinfer1::ICudaEngine *readEngine(const std::string &engineFilename);

    static TrtLogger &getLogger() {
        return trtLogger;
    }

private:
    static TrtLogger trtLogger;


};
