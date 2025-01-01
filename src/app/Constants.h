#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace Constants {
    enum class CountryCode {
        KZ,
        KG,
        UZ,
        RU,
        BY,
        GE,
        AM,
        AZ
    };


    const std::string UNIDENTIFIED_COUNTRY;
    const std::string JPG_EXTENSION = ".jpg";


    const std::string IMAGE_DIRECTORY = "/app/storage/images/";
    const std::string IMAGE_DIRECTORY_CROPPED = "/app/storage/cropped/";
    const std::string IMAGE_DIRECTORY_ORIGINAL = "/app/storage/full/";

    const std::string PLATE_IMAGE_DIR = "./images/";

    const std::string modelWeightsFolder = "./models/";


    const int DETECTION_IMG_W = 512;
    const int DETECTION_IMG_H = 512;
    const int IMG_CHANNELS = 3;
    const int PLATE_COORDINATE_SIZE = 13;
    const int DETECTION_BATCH_SIZE = 1;

    const int RECT_LP_H = 32;
    const int RECT_LP_W = 128;

    const int SQUARE_LP_H = 64;
    const int SQUARE_LP_W = 64;

    const int RECOGNIZER_MAX_BATCH_SIZE = 1;
    const int BLACK_IMG_WIDTH = 12;
    constexpr float PIXEL_MAX_VALUE = 255;

    const int LP_WHITENESS_MAX = 200;
    const int LP_WHITENESS_MIN = 90;

    const std::vector<cv::Point2f> RECT_LP_COORS{
            cv::Point2f(0, 0),
            cv::Point2f(0, 31),
            cv::Point2f(127, 0),
            cv::Point2f(127, 31),
    };

    const std::vector<cv::Point2f> SQUARE_LP_COORS{
            cv::Point2f(0, 0),
            cv::Point2f(0, 63),
            cv::Point2f(63, 0),
            cv::Point2f(63, 63),
    };

    const int CAR_MODEL_BATCH_SIZE = 1;


    enum class VehicleClassificationType {
        modelType,
        carType,
        noType,
        baqordaType
    };

    const std::unordered_map<VehicleClassificationType, std::string> CLS_TYPE_TO_WEIGHTS_NAME{
            {VehicleClassificationType::carType,     "vehicle_type_classical.np"},
            {VehicleClassificationType::baqordaType, "vehicle_type_baqorda.np"},
    };

    const std::unordered_map<VehicleClassificationType, std::string> CLS_TYPE_TO_ENGINE_NAME{
            {VehicleClassificationType::carType,     "mobilenet_v2.engine"},
            {VehicleClassificationType::baqordaType, "mobilenet_v3.engine"},
    };

    const std::unordered_map<VehicleClassificationType, int> CLS_TYPE_TO_IMG_WIDTH{
            {VehicleClassificationType::modelType,   320},
            {VehicleClassificationType::carType,     224},
            {VehicleClassificationType::baqordaType, 224},
    };

    const std::unordered_map<VehicleClassificationType, int> CLS_TYPE_TO_IMG_HEIGHT{
            {VehicleClassificationType::modelType,   320},
            {VehicleClassificationType::carType,     224},
            {VehicleClassificationType::baqordaType, 224},
    };

    const std::unordered_map<VehicleClassificationType, int> CLS_TYPE_TO_OUTPUT_SIZE{
            {VehicleClassificationType::modelType,   431},
            {VehicleClassificationType::carType,     5},
            {VehicleClassificationType::baqordaType, 5},
    };
    const std::unordered_map<std::string, Constants::CountryCode> STRING_TO_COUNTRY{
            {"KZ", Constants::CountryCode::KZ},
            {"KG", Constants::CountryCode::KG},
            {"UZ", Constants::CountryCode::UZ},
            {"RU", Constants::CountryCode::RU},
            {"BY", Constants::CountryCode::BY},
            {"GE", Constants::CountryCode::GE},
            {"AM", Constants::CountryCode::AM},
            {"AZ", Constants::CountryCode::AZ},
    };

}

