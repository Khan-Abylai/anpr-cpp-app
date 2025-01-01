//
// Created by kartykbayev on 6/20/23.
//

#pragma once

#include <filesystem>
#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "Package.h"
#include "../SharedQueue.h"
#include "Snapshot.h"

struct ImageInfo {
    int year;
    int month;
    int day;
    std::string time;
    std::filesystem::path baseFolderPath;
    std::filesystem::path yearFolderPath;
    std::filesystem::path monthFolderPath;
    std::filesystem::path dayFolderPath;
    std::filesystem::path imagePath;
    std::filesystem::path licensePlateImagePath;
};


class ImageWriterService : public IThreadLauncher, public ILogger {
public:
    explicit ImageWriterService(std::string cameraIp, const std::string &baseFolderPath);

    void addToQueue(const std::shared_ptr<Package> &package);
    void addToQueueSnapshot(const std::shared_ptr<Snapshot> &snapshot);

    ImageInfo extractInfo(const std::shared_ptr<Package> &package);

    ImageInfo extractInfoFromSnapshot(const std::shared_ptr<Snapshot> &snapshot);

    void run() override;

    void shutdown() override;

private:
    std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue;
    std::filesystem::path baseFolder;
    std::string CAMERA_IP;

    bool imageWrite(const std::shared_ptr<Package> &package);
    bool imageWriteSnapshot(const std::shared_ptr<Snapshot> &snapshot);
};
