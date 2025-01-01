//
// Created by kartykbayev on 8/22/22.
//
#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "../Config.h"
#include "../ITimer.h"
#include "../app/LicensePlate.h"
#include "../app/Utils.h"

class Snapshot : public ITimer {
public:
    Snapshot(std::string cameraIp, cv::Mat carImage, std::string snapshotUrl, bool useSecondary, std::string secondaryUrl);

    [[nodiscard]] std::string getPackageJsonString() const;

    [[nodiscard]] const std::string &getCameraIp() const;

    [[nodiscard]] std::string getSnapshotUrl() const;

    [[nodiscard]] bool useSecondaryUrl() const;

    [[nodiscard]] std::string getSecondarySnapshotUrl() const;

    [[nodiscard]] std::string getEventTime() const;

    std::string getLightweightPackageJsonString(std::string &imagePath) const;

    [[nodiscard]] cv::Mat getCarImage() const;

private:
    time_t eventTime;
    std::string cameraIp, snapshotUrl, secondarySnapshotUrl;
    cv::Mat carImage;
    bool useSecondary;
};
