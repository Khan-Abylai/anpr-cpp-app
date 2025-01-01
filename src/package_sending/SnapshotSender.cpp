//
// Created by kartykbayev on 8/22/22.
//

#include "SnapshotSender.h"

#include <utility>
#include <glib.h>

using namespace std;

SnapshotSender::SnapshotSender(
        std::shared_ptr<SharedQueue<std::shared_ptr<Snapshot>>> snapshotQueue,
        const std::vector<CameraScope> &cameraScope,
        bool imageWriteServiceEnabled, const std::string &baseFolder) : ILogger("Snapshot Sender -------------"),
                                                                        snapshotQueue{std::move(snapshotQueue)},
                                                                        imageWriterEnabled{imageWriteServiceEnabled} {
    for (const auto &cameraIp: cameraScope) {
        LOG_INFO("Camera ip:%s will send snapshot to:%s", cameraIp.getCameraIp().c_str(),
                 cameraIp.getSnapshotSendIp().c_str());
        if (cameraIp.isUseSecondarySnapshotEndpoint())
            LOG_INFO("Camera ip:%s will send also to another snapshot url:%s", cameraIp.getCameraIp().c_str(),
                     cameraIp.getOtherSnapshotSendEndpoint().c_str());
        else
            LOG_INFO("Camera ip:%s not defines secondary snapshot url", cameraIp.getCameraIp().c_str());
    }
    initializeImageWriterMap(cameraScope, baseFolder);
}

std::future<cpr::Response> SnapshotSender::sendRequestAsync(const string &jsonString, const std::string &url) {
    return cpr::PostAsync(cpr::Url{url}, cpr::VerifySsl(false), cpr::Body{jsonString},
                          cpr::Timeout{SEND_REQUEST_TIMEOUT}, cpr::Header{{"Content-Type", "application/json"}});
}

void SnapshotSender::run() {
    time_t beginTime = time(nullptr);

    queue<future<cpr::Response>> futureResponses;
    queue<future<cpr::Response>> futureResponsesSecondary;

    while (!shutdownFlag) {
        auto package = snapshotQueue->wait_and_pop();
        if (package == nullptr)
            continue;
        auto imagePath = getFullPath(package);
        if (imageWriterEnabled) {
            auto dataToSend = package->getLightweightPackageJsonString(imagePath);
            futureResponses.push(sendRequestAsync(dataToSend, package->getSnapshotUrl()));
            addPackageToImageWrite(package);
            if (package->useSecondaryUrl())
                futureResponsesSecondary.push(
                        sendRequestAsync(dataToSend, package->getSecondarySnapshotUrl()));
        } else {
            futureResponses.push(sendRequestAsync(package->getPackageJsonString(), package->getSnapshotUrl()));
            if (package->useSecondaryUrl())
                futureResponsesSecondary.push(
                        sendRequestAsync(package->getPackageJsonString(), package->getSecondarySnapshotUrl()));
        }
        while (futureResponses.size() > MAX_FUTURE_RESPONSES) {
            std::queue<future<cpr::Response>> empty;
            std::swap(futureResponses, empty);
            if (DEBUG)
                LOG_INFO("swapped snapshot queue with empty queue");
        }

        while (futureResponsesSecondary.size() > MAX_FUTURE_RESPONSES) {
            std::queue<future<cpr::Response>> empty;
            std::swap(futureResponsesSecondary, empty);
            if (DEBUG)
                LOG_INFO("swapped snapshot queue with empty queue");
        }
    }
}

void SnapshotSender::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    snapshotQueue->push(nullptr);
    for (auto &imageWriter: cameraIpToImageWriterMap) {
        imageWriter.second->shutdown();
    }
}

void SnapshotSender::initializeImageWriterMap(const vector<CameraScope> &cameras, const string &baseFolder) {
    for (const auto &camera: cameras) {
        auto imageWriterService = make_unique<ImageWriterService>(camera.getCameraIp(), baseFolder);
        imageWriterService->run();
        cameraIpToImageWriterMap.insert({camera.getCameraIp(), std::move(imageWriterService)});
    }
}

std::string SnapshotSender::getFullPath(const shared_ptr<Snapshot> &package) {
    string cameraIp = package->getCameraIp();
    auto event = cameraIpToImageWriterMap.find(cameraIp);
    if (event != cameraIpToImageWriterMap.end())
        return event->second->extractInfoFromSnapshot(package).imagePath;
    else
        return "";
}

void SnapshotSender::addPackageToImageWrite(const shared_ptr<Snapshot> &snapshot) {
    string cameraIp = snapshot->getCameraIp();
    auto event = cameraIpToImageWriterMap.find(cameraIp);
    if (event != cameraIpToImageWriterMap.end()) event->second->addToQueueSnapshot(snapshot);
}
