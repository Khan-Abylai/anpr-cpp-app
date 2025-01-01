//
// Created by artyk on 11/9/2023.
//
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "../app/Constants.h"
#include "../app/CameraScope.h"
#include "../app/LicensePlate.h"

class VehicleRecognizer {
public:
    VehicleRecognizer() = default;

    ~VehicleRecognizer() = default;

    virtual std::pair<std::string, double> classify(const std::shared_ptr<LicensePlate> &licensePlate);

protected:
    static std::vector<float> softmax(std::vector<float> &score_vec);

    float CAR_MODEL_RECOGNITION_THRESHOLD = 0.85;
    const int
            IMG_CHANNELS = Constants::IMG_CHANNELS;
};


