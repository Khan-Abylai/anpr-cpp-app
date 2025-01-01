#pragma once

#include <unordered_set>

#include "LicensePlate.h"
#include "Utils.h"
#include "Counter.h"
#include "../ILogger.h"

enum MotionDirection {
    FORWARD,
    REVERSE,
    UNDETERMINED
};

enum class Directions {
    initial = 0,
    forward = 1,
    reverse = -1
};

struct CounterHashFunction {
    struct Hash {
        std::size_t operator()(std::shared_ptr<Counter<std::shared_ptr<LicensePlate>>> const &p) const noexcept {
            return std::hash<std::string>{}(p->getItem()->getPlateLabel());
        }
    };

    struct Compare {
        size_t operator()(std::shared_ptr<Counter<std::shared_ptr<LicensePlate>>> const &a,
                          std::shared_ptr<Counter<std::shared_ptr<LicensePlate>>> const &b) const {
            return a->getItem()->getPlateLabel() == b->getItem()->getPlateLabel();
        }
    };
};

class Car : public ILogger, public ITimer {
public:
    explicit Car(const int &platesCount, const std::string &cameraIp, const std::string &originPoint,
                 const cv::Point2f &imageSizes);

    void addLicensePlateToCount(std::shared_ptr<LicensePlate> licensePlate);

    const std::shared_ptr<LicensePlate> &
    getMostCommonLicensePlate(bool showPlates = true, bool showModels = true) const;

    bool isLicensePlateBelongsToCar(const std::string &otherPlateLabel);

    bool isLicensePlateBelongsToCar(const std::shared_ptr<LicensePlate> &curLicensePlate,
                                    double lastFrameRTPTimestamp);

    void drawTrackingPoints(cv::Mat &image);

    static void drawBoundingBoxPoints(cv::Mat &image, const std::shared_ptr<LicensePlate> &licensePlate);

    void mostCommonCarModelsShow();

    void addTrackingPoint(const cv::Point2f &point, bool packageSent);

    bool isTrackPoint(const cv::Point2f &centerPoint);

    bool doesPlatesCollected();

    bool doesPlatesDoubleCollected();

    void addTrackingCarModel(const std::basic_string<char> &carModel, double probability);

    std::pair<std::string, std::pair<int, double>> getCarModelWithOccurrence();

    time_t getLifeTimeOfCar() const;

    Directions getDirection();

    void setBestFrame();

private:

    static Directions
    determineMotionDirection(const std::vector<cv::Point2f> &coordinates, const cv::Point2f &originPoint);

    Directions direction = Directions::initial;
    const int MIN_DISTANCE_BETWEEN_POINTS = 8;
    const int MAX_DIFFERENCE_BETWEEN_TRACKING_POINTS = 2;
    const float MAX_DIFFERENCE_BETWEEN_TIMESTAMPS = 1.0;
    static const int EDIT_DISTANCE_THRESHOLD = 1;
    int TRACK_POINTS_THRESHOLD;
    int MIN_NUM_PLATES_COLLECTED;
    constexpr static const float MULTIPLIER = 2.0;
    constexpr static const int TIMEOUT_OF_PLATE = 15;
    time_t startOfCarLife = time(nullptr);


    cv::Point2i originPointCoords{};

    std::unordered_set<std::shared_ptr<Counter<std::shared_ptr<LicensePlate>>>,
            CounterHashFunction::Hash, CounterHashFunction::Compare> licensePlates;

    std::shared_ptr<Counter<std::shared_ptr<LicensePlate>>> mostCommonPlate;
    std::vector<cv::Point2f> trackingPoints;
    std::map<std::string, std::pair<int, std::vector<double>>> trackingCarModels;

    static double getDistance(const cv::Point2f &punt1, const cv::Point2f &punt2) {
        return std::hypot(punt2.x - punt1.x, punt2.y - punt1.y);
    }

    static std::pair<std::string, std::pair<int, double>>
    findEntryWithLargestValue(std::map<std::string, std::pair<int, std::vector<double>>> &sampleMap) {
        std::pair<std::string, std::pair<int, double>> entryWithMaxValue = std::make_pair("", std::make_pair(0, 0.0));

        std::map<std::string, std::pair<int, std::vector<double>>>::iterator currentEntry;
        for (currentEntry = sampleMap.begin();
             currentEntry != sampleMap.end();
             ++currentEntry) {

            double currentEntryAverage =
                    std::reduce(currentEntry->second.second.begin(), currentEntry->second.second.end()) /
                    (int) currentEntry->second.second.size();

            if (currentEntry->second.first == entryWithMaxValue.second.first) {
                if (currentEntryAverage >= entryWithMaxValue.second.second) {
                    entryWithMaxValue = std::make_pair(
                            currentEntry->first,
                            std::make_pair(currentEntry->second.first, currentEntryAverage));
                }
            } else if (currentEntry->second.first > entryWithMaxValue.second.first) {
                entryWithMaxValue = std::make_pair(
                        currentEntry->first,
                        std::make_pair(currentEntry->second.first, currentEntryAverage));
            }
        }
        return entryWithMaxValue;
    }

};

