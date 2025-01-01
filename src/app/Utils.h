#pragma once

#include <opencv2/opencv.hpp>

#include "Constants.h"

class Utils {
public:

    struct convert {
        void operator()(char& c) { c = toupper((unsigned char)c); }
    };
    static std::string base64Encode(const std::vector<uint8_t> &input);

    static std::string dateTimeToStr(time_t dateTime, bool oldFormat=true);

    static std::string dateTimeToStrAnother(time_t dateTime);

    static std::string pointToStr(float x, float y);

    static std::string encodeImgToBase64(const cv::Mat &img, int quality = 50);

    static std::vector<std::string> splitString(const std::string &str, const std::string &delimiter);

    static int calculateEditDistance(const std::string &str1, const std::string &str2);

    static double calculateDistanceBetweenTwoPoints(const cv::Point2f &point1, const cv::Point2f &point2);

private:
    constexpr static const char kEncodeLookup[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

};
