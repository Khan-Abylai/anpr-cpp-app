#include "Car.h"

using namespace std;

Car::Car(const int &platesCount, const std::string &cameraIp, const std::string &originPoint,
         const cv::Point2f &imageSizes) : ILogger(
        "Car " + cameraIp),
                                          MIN_NUM_PLATES_COLLECTED{
                                                  platesCount}, TRACK_POINTS_THRESHOLD{platesCount} {
    direction = Directions::initial;

    if (originPoint == "left")
        originPointCoords = cv::Point2i{0, (int) imageSizes.y};
    else if (originPoint == "center")
        originPointCoords = cv::Point2i{(int) imageSizes.x / 2, (int) imageSizes.y};
    else if (originPoint == "right")
        originPointCoords = cv::Point2i{(int) imageSizes.x, (int) imageSizes.y};
    else
        originPointCoords = cv::Point2i{0, 0};

    startOfCarLife = time(nullptr);
    auto startTime = chrono::high_resolution_clock::now();
    setStartTime(startTime);
}

void Car::addLicensePlateToCount(shared_ptr<LicensePlate> licensePlate) {
    auto newLicensePlate = make_shared<Counter<shared_ptr<LicensePlate>>>(std::move(licensePlate));
    auto licensePlateIterator = licensePlates.find(newLicensePlate);
    if (licensePlateIterator != licensePlates.end()) {
        const auto &foundLicensePlate = *licensePlateIterator;
        foundLicensePlate->incrementOccurrence();
        foundLicensePlate->renewItem(newLicensePlate->getItem());

        if (!mostCommonPlate || mostCommonPlate->getNumberOfOccurrence() < foundLicensePlate->getNumberOfOccurrence())
            mostCommonPlate = foundLicensePlate;
    } else {
        if (!mostCommonPlate)
            mostCommonPlate = newLicensePlate;
        licensePlates.insert({newLicensePlate});
    }
}

bool Car::isTrackPoint(const cv::Point2f &centerPoint) {
    if (trackingPoints.empty())
        return true;

    auto distance = Utils::calculateDistanceBetweenTwoPoints(trackingPoints.back(), centerPoint);
    return distance >= MIN_DISTANCE_BETWEEN_POINTS;
}

bool Car::isLicensePlateBelongsToCar(const shared_ptr<LicensePlate> &curLicensePlate, double lastFrameRTPTimestamp) {
    double minDistance = INT_MAX;
    int trackingIndex;

    for (int i = (int) trackingPoints.size() - 1; i >= 0; i--) {
        auto distance = Utils::calculateDistanceBetweenTwoPoints(trackingPoints[i], curLicensePlate->getCenter());
        if (minDistance > distance) {
            minDistance = distance;
            trackingIndex = i;
            if (minDistance <= MIN_DISTANCE_BETWEEN_POINTS)
                break;
        }
    }
    double timeDiffBetweenFrames = curLicensePlate->getRTPtimestamp() - lastFrameRTPTimestamp;

    if (trackingPoints.size() - trackingIndex <= MAX_DIFFERENCE_BETWEEN_TRACKING_POINTS &&
        timeDiffBetweenFrames < MAX_DIFFERENCE_BETWEEN_TIMESTAMPS)
        return true;

    auto licensePlateBelongsToCar = isLicensePlateBelongsToCar(curLicensePlate->getPlateLabel());

    if ( licensePlateBelongsToCar && timeDiffBetweenFrames >= TIMEOUT_OF_PLATE)
        return false;

    return licensePlateBelongsToCar;

}

void Car::addTrackingPoint(const cv::Point2f &point, bool packageSent) {
    if (isTrackPoint(point)) {
        trackingPoints.push_back(point);
        if (trackingPoints.size() == 1) {
            this->direction = Directions::initial;
        } else if (trackingPoints.size() % TRACK_POINTS_THRESHOLD == 0 && !packageSent) {
            this->direction = determineMotionDirection(trackingPoints, originPointCoords);
        }
    }
}

bool Car::isLicensePlateBelongsToCar(const string &otherPlateLabel) {
    int distance = Utils::calculateEditDistance(mostCommonPlate->getItem()->getPlateLabel(), otherPlateLabel);
    return distance <= EDIT_DISTANCE_THRESHOLD;
}


const std::shared_ptr<LicensePlate> &Car::getMostCommonLicensePlate(bool showPlates, bool showModels) const {
    if (showPlates) {
        LOG_INFO("---- found plates -----");
        for (const auto &plate: licensePlates) {
            LOG_INFO("plate: %s, count: %d", plate->getItem()->getPlateLabel().data(), plate->getNumberOfOccurrence());
        }
    }

    if (showModels) {
        LOG_INFO("---- found models -----");
        for (const auto &model: trackingCarModels) {
            auto tempItem =
                    std::reduce(model.second.second.begin(), model.second.second.end()) / model.second.second.size();
            LOG_INFO("model: %s, count: %d probability:%f", model.first.c_str(), model.second.first,
                     tempItem);
        }
    }

    return mostCommonPlate->getItem();;
}

void Car::setBestFrame(){
    double maxProb = 0.0;
    cv::Mat bestFrame;
    for(const auto &plate: licensePlates) {
        if (plate->getItem()->getPlateProbability() > maxProb){
            maxProb = plate->getItem()->getPlateProbability();
            bestFrame = plate->getItem()->getCarImage();
            plate->getItem()->setCarImage(bestFrame);
        }
    }
}

void Car::drawTrackingPoints(cv::Mat &image) {
    for (const auto &centerPoint: trackingPoints) {
        cv::circle(image, centerPoint, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_8);
    }
}

void Car::drawBoundingBoxPoints(cv::Mat &image, const shared_ptr<LicensePlate> &licensePlate) {
    cv::circle(image, licensePlate->getLeftTop(), 3, cv::Scalar(0, 255, 0));
    cv::circle(image, licensePlate->getLeftBottom(), 3, cv::Scalar(0, 255, 0));
    cv::circle(image, licensePlate->getRightTop(), 3, cv::Scalar(0, 255, 0));
    cv::circle(image, licensePlate->getRightBottom(), 3, cv::Scalar(0, 255, 0));
}

Directions Car::getDirection() {
    return direction;
}


bool Car::doesPlatesCollected() {
    int platesCollected = 0;
    for (const auto &plate: licensePlates) platesCollected += plate->getNumberOfOccurrence();

    if (platesCollected >= MIN_NUM_PLATES_COLLECTED) return true;
    return false;
}

void Car::addTrackingCarModel(const basic_string<char> &carModel, const double probability) {
    if (trackingCarModels.find(carModel) == trackingCarModels.end())
        trackingCarModels.insert(make_pair(carModel,
                                           make_pair(1, std::vector<double>{probability})));
    else {
        trackingCarModels[carModel].first++;
        trackingCarModels[carModel].second.push_back(probability);
    };
}


std::pair<std::string, std::pair<int, double>> Car::getCarModelWithOccurrence() {
    if (findEntryWithLargestValue(trackingCarModels).first == trackingCarModels.end()->first) {
        return pair<std::string, std::pair<int, double>>{"NotDefined", {0, 0.0}};
    } else {
        return pair<std::string, std::pair<int, double>>{findEntryWithLargestValue(trackingCarModels).first,
                                                         findEntryWithLargestValue(trackingCarModels).second};
    }
}

void Car::mostCommonCarModelsShow() {
    if (findEntryWithLargestValue(trackingCarModels).first == trackingCarModels.end()->first) {
        LOG_INFO("----- no found models -----");
    } else {
        LOG_INFO("---- found models -----");
        for (const auto &model: trackingCarModels) {
            auto tempItem =
                    std::reduce(model.second.second.begin(), model.second.second.end()) / model.second.second.size();
            LOG_INFO("model: %s, count: %d probability: %f", model.first.c_str(), model.second.first,
                     tempItem);
        }
    }
}

bool Car::doesPlatesDoubleCollected() {
    int platesCollected = 0;
    for (const auto &plate: licensePlates) platesCollected += plate->getNumberOfOccurrence();

    if (platesCollected * 1.0 >= MIN_NUM_PLATES_COLLECTED * 1.0 * MULTIPLIER) return true;
    return false;
}

Directions Car::determineMotionDirection(const vector<cv::Point2f> &coordinates, const cv::Point2f &originPoint) {
    const cv::Point2f &startPoint = coordinates.front();
    const cv::Point2f &endPoint = coordinates.back();

    cv::Point2f startPtOriginPoint = {startPoint.x, originPoint.y};
    cv::Point2f endPtOriginPoint = {endPoint.x, originPoint.y};
    auto deltaA = getDistance(startPoint, startPtOriginPoint);
    auto deltaB = getDistance(endPoint, endPtOriginPoint);

    if (deltaB < deltaA)
        return Directions::forward;
    else
        return Directions::reverse;
}

time_t Car::getLifeTimeOfCar() const {
    return startOfCarLife;
}

