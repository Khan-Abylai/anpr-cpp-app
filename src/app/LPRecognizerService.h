#pragma once

#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "../RandomStringGenerator.h"
#include "LPRecognizer.h"
#include "CalibParams.h"
#include "TemplateMatching.h"
#include "../package_sending/Package.h"
#include "BaseCarTracker.h"
#include "MaskCarTracker.h"
#include "LicensePlate.h"
#include "Constants.h"
#include "CameraScope.h"

using namespace std;

class LPRecognizerService : public IThreadLauncher, public ::ILogger {
public:
    LPRecognizerService(const std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> &packageQueue,
                        const std::vector<CameraScope> &cameras, float recognizerThreshold,
                        std::pair<float, float> calibrationSizes, int statInterval,
                        const std::unordered_map<std::string, Constants::CountryCode>& templates);


    void addToQueue(std::shared_ptr<LicensePlate> licensePlate);

    void run() override;

    void shutdown() override;

    CameraScope getCameraScope(const std::string &cameraIp) const;

private:
    const std::vector<CameraScope> &cameraScopeVector;
    float RECOGNIZER_THRESHOLD;
    std::string NODE_IP;
    cv::Mat whiteImage = cv::Mat(Constants::RECT_LP_H, Constants::RECT_LP_W, CV_8UC3, cv::Scalar(255, 255, 255));

    std::unique_ptr<SharedQueue<std::shared_ptr<LicensePlate>>> lpQueue;
    std::unordered_map<std::string, std::unique_ptr<BaseCarTracker>> cameraIpToCarTrackerMap;
    std::unique_ptr<LPRecognizer> lpRecognizer;
    std::unique_ptr<TemplateMatching> templateMatching;

    unordered_map<string, int> cameraToNotValidByPatternPlatesCount;
    unordered_map<string, int> cameraToNotValidByConfPlatesCount;
    unordered_map<string, int> cameraToNotValidBothPlatesCount;
    unordered_map<string, int> cameraToOverallPlatesCount;
    unordered_map<string, int> cameraToPassPlatesCount;

    unordered_map<string, double> cameraIpToPlateWhitenessSum;
    unordered_map<string, double> cameraIpToPlateWhitenessNum;
    unordered_map<string, double> cameraIpToPlateDFTSharpnessSum;
    unordered_map<string, double> cameraIpToPlateLaplacianSharpnessSum;
    unordered_map<string, double> cameraIpToWhitePlatesNum;
    unordered_map<string, double> cameraIpToDarkPlatesNum;
    unordered_map<string, double> cameraIpToCurrentCameraWhiteness;

    unordered_map<string, vector<double>> cameraToNotValidProbs;
    unordered_map<string, vector<double>> cameraToValidProbs;

    std::condition_variable shutdownEvent;
    std::mutex checkCurrentAuto;
    std::thread backgroundThread;
    int waitMinutes{};

    std::pair<std::string, float>
    getLicensePlateLabel(const std::vector<std::pair<std::string, float>> &recognizerResult, bool isSquarePlate);


    bool isValidLicensePlate(const std::string &lpLabel, float probability);

    bool isValidByConf(float probability) const;

    bool isValidByPattern(const std::string &lpLabel);

    void addPlateToTrack(const std::shared_ptr<LicensePlate> &licensePlate);

    static bool isChooseThisFrame();

    static void saveFrame(const std::shared_ptr<LicensePlate> &plate);

    static std::vector<cv::Mat> getLicensePlateImages(const std::shared_ptr<LicensePlate> &licensePlate);

    void initializeCameraIpCarTrackerMap(const std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> &packageQueue,
                                         const std::vector<CameraScope> &cameras,
                                         std::pair<float, float> calibrationSizes);

    void checkCurrentPlatesStatus();

    static double mean(std::vector<double> &nums);

    static double median(std::vector<double> &nums);

    static cv::Mat combineInOneLineSquareLpPlate(const cv::Mat &lpImage);

    static std::vector<std::string> string_split(const std::string &s, const char delimiter) {
        size_t start = 0;
        size_t end = s.find_first_of(delimiter);

        std::vector<std::string> output;

        while (end <= std::string::npos) {
            output.emplace_back(s.substr(start, end - start));

            if (end == std::string::npos)
                break;

            start = end + 1;
            end = s.find_first_of(delimiter, start);
        }

        return output;
    }

};
