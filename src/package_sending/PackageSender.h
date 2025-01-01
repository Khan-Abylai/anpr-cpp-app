#pragma once

#include <ctime>
#include <future>
#include <unordered_map>

#include <cpr/cpr.h>

#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "../app/LicensePlate.h"
#include "Package.h"
#include "../app/Utils.h"
#include "../app/CameraScope.h"
#include "ImageWriterService.h"

class PackageSender : public IThreadLauncher, public ILogger {
public:
    PackageSender(std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                  const std::vector<CameraScope> &cameraScope, const std::string &baseFolder,
                  bool imageWriteServiceEnabled);

    void run() override;

    void shutdown() override;

private:

    void initializeImageWriterMap(const std::vector<CameraScope> &cameras, const std::string &baseFolder);

    void addPackageToImageWrite(const std::shared_ptr<Package> &package);

    std::pair<std::string, std::string> getPairOfImagePaths(const std::shared_ptr<Package> &package);


    const int SEND_REQUEST_TIMEOUT = 10000;
    const int MAX_FUTURE_RESPONSES = 30;

    bool imageWriterEnabled = false;

    std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue;

    std::unordered_map<std::string, std::unique_ptr<ImageWriterService>> cameraIpToImageWriterMap;


    std::future<cpr::Response> sendRequestAsync(const std::string &jsonString, const std::string &url);

    double getAverageTime(double averageTime, double currentTime) const;

    double avgDetectionTime = 0;
    double avgRecognizerTime = 0;
    double avgOverallTime = 0;
    double avgEventTime = 0;

    double maxDetectionTime = -1.0;
    double maxRecognizerTime = -1.0;
    double maxOverallTime = -1.0;
    double maxEventTime = -1.0;
    int iteration = 0;

    int debugTime = 300;

};