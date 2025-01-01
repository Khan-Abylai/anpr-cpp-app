#include "PackageSender.h"

#include <utility>

using namespace std;

future<cpr::Response> PackageSender::sendRequestAsync(const string &jsonString, const string &url) {
    return cpr::PostAsync(cpr::Url{url}, cpr::VerifySsl(false), cpr::Body{jsonString},
                          cpr::Timeout{SEND_REQUEST_TIMEOUT}, cpr::Header{{"Content-Type", "application/json"}});
}

PackageSender::PackageSender(shared_ptr<SharedQueue<shared_ptr<Package>>> packageQueue,
                             const vector<CameraScope> &cameraScope, const std::string &baseFolder,
                             bool imageWriteServiceEnabled) : ILogger("Package Sender --------------------"),
                                                              packageQueue{std::move(packageQueue)},
                                                              imageWriterEnabled{imageWriteServiceEnabled} {
    initializeImageWriterMap(cameraScope, baseFolder);
}

void PackageSender::run() {
    time_t beginTime = time(nullptr);

    queue<future<cpr::Response >> futureResponses;
    queue<future<cpr::Response >> futureResponsesSecondary;
    while (!shutdownFlag) {
        auto package = packageQueue->wait_and_pop();
        if (package == nullptr) continue;

        auto [imagePath, lpImagePath] = getPairOfImagePaths(package);
        if (imageWriterEnabled) {
            LOG_INFO("%s %s %s %s %s %s", package->getCameraIp().data(), package->getPlateLabel().data(),
                     Utils::dateTimeToStr(time_t(nullptr)).c_str(), imagePath.c_str(), package->getCarModel().c_str(), lpImagePath.data());
            auto dataToSend = package->getLightweightPackageJsonString(imagePath, lpImagePath);
            futureResponses.push(sendRequestAsync(dataToSend, package->getResultSendUrl()));
            addPackageToImageWrite(package);
            if (package->doesSecondaryResultSendUrlEnabled())
                futureResponsesSecondary.push(sendRequestAsync(dataToSend, package->getSecondaryResultSendUrl()));
        } else {
            LOG_INFO("%s %s %s %s", package->getCameraIp().data(), package->getPlateLabel().data(),
                     Utils::dateTimeToStr(time_t(nullptr)).c_str(), package->getCarModel().c_str());
            futureResponses.push(sendRequestAsync(package->getPackageJsonString(), package->getResultSendUrl()));
            if (package->doesSecondaryResultSendUrlEnabled())
                futureResponsesSecondary.push(
                        sendRequestAsync(package->getPackageJsonString(), package->getSecondaryResultSendUrl()));
        }


        while (futureResponses.size() > MAX_FUTURE_RESPONSES) {
            futureResponses.pop();
        }

        while (futureResponsesSecondary.size() > MAX_FUTURE_RESPONSES) {
            futureResponsesSecondary.pop();
        }


        LOG_INFO("plate: %s, det time: %lf, rec time: %lf, overall time: %lf, full time of event: %lf", package->getPlateLabel().data(), package->getDetectionTime(), package->getRecognizerTime(), package->getOverallTime(), package->getRealTimeOfEvent());


        avgDetectionTime = getAverageTime(avgDetectionTime, package->getDetectionTime());
        avgRecognizerTime = getAverageTime(avgRecognizerTime, package->getRecognizerTime());
        avgOverallTime = getAverageTime(avgOverallTime, package->getOverallTime());
        avgEventTime = getAverageTime(avgEventTime, package->getRealTimeOfEvent());

        maxDetectionTime = max(maxDetectionTime, package->getDetectionTime());
        maxRecognizerTime = max(maxRecognizerTime, package->getRecognizerTime());
        maxOverallTime = max(maxOverallTime, package->getOverallTime());
        maxEventTime = max(maxEventTime, package->getRealTimeOfEvent());
        iteration++;

        if (time(nullptr) - beginTime > debugTime) {
            beginTime = time(nullptr);
            LOG_INFO("average detection time %lf", avgDetectionTime);
            LOG_INFO("average recognizer time %lf", avgRecognizerTime);
            LOG_INFO("average overall time %lf", avgOverallTime);
            LOG_INFO("average event time %lf", avgEventTime);
            LOG_INFO("max detection time %lf", maxDetectionTime);
            LOG_INFO("max recognizer time %lf", maxRecognizerTime);
            LOG_INFO("max overall time %lf", maxOverallTime);
            LOG_INFO("max event time %lf",maxEventTime);
        }


    }
}

void PackageSender::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    packageQueue->push(nullptr);
    for (auto &imageWriter: cameraIpToImageWriterMap) {
        imageWriter.second->shutdown();
    }
}

void PackageSender::initializeImageWriterMap(const vector<CameraScope> &cameras, const string &baseFolder) {
    for (const auto &camera: cameras) {
        auto imageWriterService = make_unique<ImageWriterService>(camera.getCameraIp(), baseFolder);
        imageWriterService->run();
        cameraIpToImageWriterMap.insert({camera.getCameraIp(), std::move(imageWriterService)});
    }

}

void PackageSender::addPackageToImageWrite(const shared_ptr<Package> &package) {
    string cameraIp = package->getCameraIp();
    auto event = cameraIpToImageWriterMap.find(cameraIp);
    if (event != cameraIpToImageWriterMap.end()) event->second->addToQueue(package);
}


std::pair<std::string, std::string> PackageSender::getPairOfImagePaths(const shared_ptr<Package> &package) {
    ;
    string cameraIp = package->getCameraIp();
    auto event = cameraIpToImageWriterMap.find(cameraIp);
    if (event != cameraIpToImageWriterMap.end())
        return std::make_pair(event->second->extractInfo(package).imagePath,
                              event->second->extractInfo(package).licensePlateImagePath);
    else
        return {};
}

double PackageSender::getAverageTime(double averageTime, double currentTime) const {
    return averageTime * (static_cast<double>(iteration) / (iteration + 1)) + currentTime / (iteration + 1);
}


