#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "app/CameraScope.h"
#include "app/Utils.h"

class Config {
public:

    static bool parseJson(const std::string &fileName);

    static const std::vector<std::vector<CameraScope>> &getCameras();

    static bool useGPUDecode();

    static const float &getTimeSentSnapshots();

    static const float &getRecognizerThreshold();

    static const float &getDetectorThreshold();

    static std::pair<float, float> getCalibrationSizes();

    static const std::string &getBaseFolder();

    static bool useImageWriter();

    static int &getStatInterval();

    static const std::vector<CameraScope> &getAllCameras();

private:

    static std::vector<std::vector<CameraScope>> reshapeVector(const std::vector<CameraScope> &input , int n);
    static std::vector<float> splitStringToFloats(const std::string &input) {
        std::vector<float> values;
        std::stringstream ss(input);
        std::string token;

        while (getline(ss, token, ',')) {
            float value = std::stof(token);
            values.push_back(value);
        }

        return values;
    }

    static bool validateString(const std::string &input) {
        std::vector<std::string> values;
        std::stringstream ss(input);
        std::string token;

        while (getline(ss, token, ',')) {
            if (token.empty()) {
                // Empty token indicates a missing number
                return false;
            }
            values.push_back(token);
        }

        return values.size() == 4;
    }


};