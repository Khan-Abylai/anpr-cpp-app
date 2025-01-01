#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "../app/Utils.h"
#include "../app/LicensePlate.h"
#include "../ITimer.h"
#include "../Config.h"

class Package : public ITimer {
public:
    Package(std::string cameraIp, std::string licensePlateLabel, cv::Mat carImage, cv::Mat plateImage,
            std::string strBoundingBox, std::string carModel, std::basic_string<char> direction,
            std::string resultSendUrlParam, std::string secondaryResultSendUrlParam,
            bool secondaryResultSendUrlFlagParam, double realTimeOfEvent);

    [[nodiscard]] std::string getPackageJsonString() const;

    [[nodiscard]] const std::string &getPlateLabel() const;

    [[nodiscard]] const std::string &getCameraIp() const;

    static std::string convertBoundingBoxToStr(const std::shared_ptr<LicensePlate> &licensePlate);

    [[nodiscard]] const std::string &getCarModel() const;

    [[nodiscard]] const std::string &getResultSendUrl() const;

    [[nodiscard]] const std::string &getSecondaryResultSendUrl() const;

    [[nodiscard]] bool doesSecondaryResultSendUrlEnabled() const;

    [[nodiscard]] std::string getEventTime() const;

    [[nodiscard]] cv::Mat getCarImage() const;

    [[nodiscard]] cv::Mat getPlateImage() const;

    std::string getLightweightPackageJsonString(std::string &imagePath, std::string &licensePlateImagePath) const;

    [[nodiscard]] double getRealTimeOfEvent() const;
private:
    std::string resultSendUrl, secondaryResultSendUrl;
    bool secondaryResultSendUrlFlag;
    std::string direction;
    std::string carModel;
    std::string cameraIp;
    time_t eventTime;
    std::string licensePlateLabel;
    cv::Mat carImage;
    cv::Mat plateImage;
    std::string strBoundingBox;
    double realTimeOfEvent;
};
