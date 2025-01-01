#pragma once

#include <string>
#include "NvInfer.h"

class TensorRTEngine {

public:
    static void serializeEngine(nvinfer1::ICudaEngine *engine, const std::string &engineFilename);

    static nvinfer1::ICudaEngine *readEngine(const std::string &engineFilename);

};
