#include "Package.h"

#include <utility>

using namespace std;
using json = nlohmann::json;

Package::Package(string cameraIp, string licensePlateLabel, cv::Mat carImage, cv::Mat plateImage, string strBoundingBox,
                 basic_string<char> direction, std::string resultSendUrlParam,
                 std::string secondaryResultSendUrlParam, bool secondaryResultSendUrlFlagParam, double realTimeOfEvent)
        : resultSendUrl{
        std::move(resultSendUrlParam)}, secondaryResultSendUrl{std::move(secondaryResultSendUrlParam)},
          secondaryResultSendUrlFlag{
                  secondaryResultSendUrlFlagParam},
          direction(std::move(
                  direction)),
          cameraIp(std::move(
                  cameraIp)),
          licensePlateLabel(
                  std::move(
                          licensePlateLabel)),
          carImage(std::move(
                  carImage)),
          plateImage(std::move(
                  plateImage)),
          strBoundingBox(
                  std::move(
                          strBoundingBox)), realTimeOfEvent{realTimeOfEvent} {
    eventTime = time_t(nullptr);
};

string Package::convertBoundingBoxToStr(const shared_ptr<LicensePlate> &licensePlate) {
    auto frameSize = licensePlate->getCarImageSize();
    auto frameWidth = (float) frameSize.width;
    auto frameHeight = (float) frameSize.height;
    return Utils::pointToStr(licensePlate->getLeftTop().x / frameWidth, licensePlate->getLeftTop().y / frameHeight) +
           ", " + Utils::pointToStr(licensePlate->getLeftBottom().x / frameWidth,
                                    licensePlate->getLeftBottom().y / frameHeight) + ", " +
           Utils::pointToStr(licensePlate->getRightTop().x / frameWidth, licensePlate->getRightTop().y / frameHeight) +
           ", " + Utils::pointToStr(licensePlate->getRightBottom().x / frameWidth,
                                    licensePlate->getRightBottom().y / frameHeight);
}

string Package::getPackageJsonString() const {
    json packageJson;
    packageJson["ip_address"] = cameraIp;
    packageJson["event_time"] = Utils::dateTimeToStr(eventTime);
    packageJson["car_number"] = licensePlateLabel;
    packageJson["car_picture"] = Utils::encodeImgToBase64(carImage, 25);
    packageJson["lp_picture"] = Utils::encodeImgToBase64(plateImage, 90);
    packageJson["lp_rect"] = strBoundingBox;
    packageJson["car_model"] = "NotDefined";
    packageJson["direction"] = direction;

    return packageJson.dump();
}


const string &Package::getPlateLabel() const {
    return licensePlateLabel;
}

const string &Package::getCameraIp() const {
    return cameraIp;
}

const std::string &Package::getResultSendUrl() const {
    return resultSendUrl;
}

const std::string &Package::getSecondaryResultSendUrl() const {
    return secondaryResultSendUrl;
}

bool Package::doesSecondaryResultSendUrlEnabled() const {
    return secondaryResultSendUrlFlag;
}

string Package::getEventTime() const {
    return Utils::dateTimeToStr(eventTime);
}

cv::Mat Package::getCarImage() const {
    return carImage;
}

std::string Package::getLightweightPackageJsonString(string &imagePath, string &licensePlateImagePath) const {
    json packageJson;
    packageJson["ip_address"] = cameraIp;
    packageJson["event_time"] = Utils::dateTimeToStr(eventTime, false);
    packageJson["car_number"] = licensePlateLabel;
    packageJson["lp_path"] = licensePlateImagePath;
    packageJson["image_path"] = imagePath;
    packageJson["car_model"] = "NotDefined";
    packageJson["direction"] = direction;

    return packageJson.dump();
}

cv::Mat Package::getPlateImage() const {
    return plateImage;
}

double Package::getRealTimeOfEvent() const {
    return realTimeOfEvent;
}



