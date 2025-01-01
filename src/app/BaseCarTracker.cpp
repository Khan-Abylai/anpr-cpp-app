#include "BaseCarTracker.h"

using namespace std;

BaseCarTracker::BaseCarTracker(shared_ptr<SharedQueue<shared_ptr<Package>>> packageQueue,
                               const string &cameraIp)
        : ILogger("Car Tracker " + cameraIp),
          packageQueue{std::move(packageQueue)} {}

void BaseCarTracker::showResult(const shared_ptr<LicensePlate> &licensePlate) {
    if (!currentCar) return;

    cv::Mat copyImage;
    licensePlate->getCarImage().copyTo(copyImage);
    currentCar->drawBoundingBoxPoints(copyImage, licensePlate);
    currentCar->drawTrackingPoints(copyImage);
    cv::Mat displayImage;
    cv::resize(copyImage, displayImage,cv::Size(copyImage.cols * 0.7,copyImage.rows * 0.7), 0, 0);
    cv::imshow(licensePlate->getCameraIp(), displayImage);
    cv::waitKey(60);
}

shared_ptr<Car> BaseCarTracker::createNewCar(const int &platesCount, const std::string &cameraIp, const std::string &originPoint, const cv::Point2f &imageSizes) {
    return make_shared<Car>(platesCount, cameraIp, originPoint, imageSizes);
}

shared_ptr<Package> BaseCarTracker::createPackage(const shared_ptr<LicensePlate> &licensePlate) {
    string strBoundingBox = Package::convertBoundingBoxToStr(licensePlate);

    auto package = make_shared<Package>(licensePlate->getCameraIp(), licensePlate->getPlateLabel(),
                                        licensePlate->getCarImage(), licensePlate->getPlateImage(), strBoundingBox,
                                        licensePlate->getCarModel(), licensePlate->getDirection(),
                                        licensePlate->getResultSendUrl(), licensePlate->getSecondaryResultSendUrl(),
                                        licensePlate->doesSecondaryUrlEnabled(), licensePlate->getRealTimeOfEvent());


    package->setStartTime(licensePlate->getStartTime());
    package->setOverallTime(licensePlate->getOverallTime());
    package->setDetectionTime(licensePlate->getDetectionTime());
    package->setRecognizerTime(licensePlate->getRecognizerTime());
    return package;
}

void BaseCarTracker::createAndPushPackage(const shared_ptr<LicensePlate> &licensePlate) {
    auto package = createPackage(licensePlate);
    packageQueue->push(std::move(package));
}