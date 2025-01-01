//
// Created by kartykbayev on 8/22/22.
//
#pragma once

#include <ctime>
#include <future>
#include <unordered_map>
#include <utility>
#include <glib.h>
#include <cpr/cpr.h>

#include "../ILogger.h"
#include "../IThreadLauncher.h"
#include "../SharedQueue.h"
#include "../app/LicensePlate.h"
#include "Package.h"
#include "Snapshot.h"
#include "../app/Utils.h"
#include "../app/CameraScope.h"
#include "ImageWriterService.h"

class SnapshotSender : public IThreadLauncher, public ::ILogger {
public:
    SnapshotSender(
            std::shared_ptr<SharedQueue<std::shared_ptr<Snapshot>>> snapshotQueue,
            const std::vector<CameraScope> &cameras, bool imageWriteServiceEnabled, const std::string &baseFolder);

    void run() override;

    void shutdown() override;

private:
    const int SEND_REQUEST_TIMEOUT = 1000;
    const int MAX_FUTURE_RESPONSES = 30;
    time_t lastFrameRTPTimestamp = time(nullptr);
    time_t lastTimeSnapshotSent = time(nullptr);
    int iteration = 0;
    bool imageWriterEnabled = false;
    std::shared_ptr<SharedQueue<std::shared_ptr<Snapshot>>> snapshotQueue;

    cpr::AsyncResponse sendRequests(const std::string &jsonString, const std::string &serverUrl);

    std::unordered_map<std::string, std::unique_ptr<ImageWriterService>> cameraIpToImageWriterMap;

    void initializeImageWriterMap(const std::vector<CameraScope> &cameras, const std::string &baseFolder);

    std::string getFullPath(const std::shared_ptr<Snapshot> &package);

    void addPackageToImageWrite(const std::shared_ptr<Snapshot> &snapshot);
};
