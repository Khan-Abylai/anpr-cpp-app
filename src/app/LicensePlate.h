#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>

#include "Constants.h"
#include "Utils.h"
#include "../ITimer.h"

class LicensePlate : public ITimer {
public:

    LicensePlate(int x, int y, int w, int h, int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, float detProb);

    [[nodiscard]] const cv::Point2i &getCenter() const;

    [[nodiscard]] const cv::Point2f &getLeftTop() const;

    [[nodiscard]] const cv::Point2f &getRightBottom() const;

    [[nodiscard]] const cv::Point2f &getLeftBottom() const;

    [[nodiscard]] const cv::Point2f &getRightTop() const;

    [[nodiscard]] bool isSquare() const;

    [[nodiscard]] float getArea() const;

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[nodiscard]] cv::Size getCarImageSize() const;

    [[nodiscard]] const cv::Mat &getPlateImage() const;

    void setPlateImage(const cv::Mat &frame);

    [[nodiscard]] const std::string &getPlateLabel() const;

    void setLicensePlateLabel(std::string lpLabel);

    [[nodiscard]] const std::string &getCameraIp() const;

    void setCameraIp(std::string ip);

    [[nodiscard]] const cv::Mat &getCarImage() const;

    void setCarImage(cv::Mat image);

    void setRTPtimestamp(double timestamp);

    [[nodiscard]] double getRTPtimestamp() const;

    void setDirection(std::string direction);

    [[nodiscard]] const std::string &getDirection() const;

    [[nodiscard]] const std::string &getResultSendUrl() const;

    void setResultSendUrl(const std::string &url);

    [[nodiscard]] const std::string &getSecondaryResultSendUrl() const;

    void setSecondaryResultSendUrl(const std::string &url);

    [[nodiscard]] bool doesSecondaryUrlEnabled() const;

    void setSecondaryUrlEnabledFlag(bool flag);

    [[nodiscard]] double getSharpness() const ;

    [[nodiscard]] double getDFTSharpness() const ;

    [[nodiscard]] double getWhiteness() const;

    void setPlateProbability(double probability);

    [[nodiscard]] double getPlateProbability() const;

    void setRealTimeOfEvent(double time);

    [[nodiscard]] double getRealTimeOfEvent() const;

private:

    static double calculateWhiteScore(const cv::Mat &imageCrop);

    static double calculateSharpness(const cv::Mat &licensePlateImg);

    static double calculateBlurCoefficient(const cv::Mat &image);

    const int CROP_PADDING = 1;
    const float SQUARE_LP_RATIO = 2.6;

    cv::Mat plateImage;
    cv::Mat carImage;
    cv::Mat croppedPlateImage;

    cv::Point2i center;
    cv::Point2f leftTop;
    cv::Point2f leftBottom;
    cv::Point2f rightTop;
    cv::Point2f rightBottom;

    std::string licensePlateLabel, cameraIp, direction, resultSendUrl, secondaryResultSendUrl;

    bool secondarySendUrlEnabled{};

    double rtpTimestamp{};
    double realTimeOfPackage;
    int width, height;
    bool square = false;

    double laplacianValue{};
    double dftValue{};
    double qualityValue{};
    double whitenessValue{};
    double plateProbability;
};