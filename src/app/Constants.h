#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace Constants {
    enum class CountryCode {
        MA,
        TA
    };

    const std::string UNIDENTIFIED_COUNTRY;
    const std::string JPG_EXTENSION = ".jpg";


    const std::string IMAGE_DIRECTORY = "/app/storage/images/";

    const std::string modelWeightsFolder = "./models/";

    const int DETECTION_IMG_W = 512;
    const int DETECTION_IMG_H = 512;
    const int IMG_CHANNELS = 3;
    const int PLATE_COORDINATE_SIZE = 13;
    const int DETECTION_BATCH_SIZE = 1;

    const int RECT_LP_H = 32;
    const int RECT_LP_W = 128;

    const int EXTENDED_RECT_LP_H = 64;
    const int EXTENDED_RECT_LP_W = 160;

    const int SQUARE_LP_H = 64;
    const int SQUARE_LP_W = 64;

    const int EXTENDED_SQUARE_LP_H = 160;
    const int EXTENDED_SQUARE_LP_W = 160;

    const int EU_LP_H = 64;
    const int EU_LP_W = 160;
    const int EU_SEQUENCE_SIZE = 38;


    const int RECOGNIZER_MAX_BATCH_SIZE = 1;
    const int BLACK_IMG_WIDTH = 12;
    constexpr float PIXEL_MAX_VALUE = 255;

    const int LP_WHITENESS_MAX = 200;
    const int LP_WHITENESS_MIN = 90;

    const std::vector<cv::Point2f> RECT_LP_COORS{
            cv::Point2f(0, 0),
            cv::Point2f(0, 63),
            cv::Point2f(159, 0),
            cv::Point2f(159, 63),
    };

    const std::vector<cv::Point2f> SQUARE_LP_COORS{
            cv::Point2f(0, 0),
            cv::Point2f(0, 63),
            cv::Point2f(63, 0),
            cv::Point2f(63, 63),
    };

    const std::vector<cv::Point2f> EXTENDED_SQUARE_LP_COORS{
        cv::Point2f(0, 0),
        cv::Point2f(0, 159),
        cv::Point2f(159, 0),
        cv::Point2f(159, 159),
    };

    const std::vector<cv::Point2f> EXTENDED_RECT_LP_COORS{
        cv::Point2f(0, 0),
        cv::Point2f(0, 63),
        cv::Point2f(159, 0),
        cv::Point2f(159, 63),
    };

    const std::unordered_map<std::string, Constants::CountryCode> STRING_TO_COUNTRY{
            {"MA", Constants::CountryCode::MA},
            {"TA", Constants::CountryCode::TA},
    };

}

