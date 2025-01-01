#include "ImageWriterService.h"

#include <utility>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::filesystem;

ImageWriterService::ImageWriterService(string cameraIp, const std::string &baseFolderPath) : ILogger(
        "ImageWriter Service"),CAMERA_IP(std::move(cameraIp)),baseFolder{baseFolderPath} {
    packageQueue = make_shared<SharedQueue<shared_ptr<Package>>>();
}


void ImageWriterService::addToQueueSnapshot(const shared_ptr<Snapshot> &snapshot) {
    if (!fs::exists(baseFolder)) {
        fs::create_directory(baseFolder);
    }
    bool success = imageWriteSnapshot(snapshot);
    if (!success) {
        LOG_ERROR("Snapshot Image was not stored on the disk");
    }
}


void ImageWriterService::addToQueue(const shared_ptr<Package> &package) {

    if (!fs::exists(baseFolder)) {
        fs::create_directory(baseFolder);
    }
    auto start = std::chrono::high_resolution_clock::now();
    bool success = imageWrite(package);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    auto in_seconds = elapsed.count() * 1e-9;
    if (!success) {
        LOG_ERROR("Image was not stored on the disk");
    }
}

void ImageWriterService::run() {
    // Implement the run function logic here
}

void ImageWriterService::shutdown() {
    LOG_INFO("Service shutting down");
    shutdownFlag = true;
    packageQueue->push(nullptr);
}

ImageInfo ImageWriterService::extractInfo(const shared_ptr<Package> &package) {
    std::tm tm = {};
    std::istringstream iss(package->getEventTime());
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    int year = tm.tm_year + 1900;
    int month = tm.tm_mon + 1;
    int day = tm.tm_mday;

    std::ostringstream oss;
    oss << std::put_time(&tm, "%H-%M-%S");
    std::string timeStr = oss.str();

    string eventFolder = "event";

    ImageInfo dt;
    dt.year = year;
    dt.month = month;
    dt.day = day;
    dt.time = timeStr;
    dt.baseFolderPath = baseFolder / eventFolder;
    dt.yearFolderPath = dt.baseFolderPath / to_string(year);
    dt.monthFolderPath = dt.yearFolderPath / to_string(month);
    dt.dayFolderPath = dt.monthFolderPath / to_string(day);
    std::string baseName = package->getCameraIp() + "_" + package->getPlateLabel() + "_" + timeStr + ".jpeg";
    std::string lpImageBaseName = package->getCameraIp() + "_lp" + ".jpeg";
    fs::path fullPath = dt.dayFolderPath / baseName;
    fs::path lpImageFullPath = dt.dayFolderPath / lpImageBaseName;

    dt.imagePath = fullPath;
    dt.licensePlateImagePath = lpImageFullPath;

    return dt;
}

bool ImageWriterService::imageWriteSnapshot(const shared_ptr<Snapshot> &snapshot) {
    auto currentFullDate = snapshot->getEventTime();
    auto imageInfo = extractInfoFromSnapshot(snapshot);

    if (!filesystem::exists(imageInfo.dayFolderPath))
        fs::create_directory(imageInfo.dayFolderPath);
    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 25};
    bool success = cv::imwrite(imageInfo.imagePath, snapshot->getCarImage(), compression_params);
    return success;
}


bool ImageWriterService::imageWrite(const shared_ptr<Package> &package) {
    auto currentFullDate = package->getEventTime();
    auto imageInfo = extractInfo(package);


    if (!filesystem::exists(imageInfo.dayFolderPath)) {
        fs::create_directory(imageInfo.baseFolderPath);
        fs::create_directory(imageInfo.yearFolderPath);
        fs::create_directory(imageInfo.monthFolderPath);
        fs::create_directory(imageInfo.dayFolderPath);
    }


    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 25};
    bool success1 = cv::imwrite(imageInfo.imagePath, package->getCarImage(), compression_params);
    bool success2 = cv::imwrite(imageInfo.licensePlateImagePath, package->getPlateImage(), compression_params);
    return success1 && success2;
}

ImageInfo ImageWriterService::extractInfoFromSnapshot(const shared_ptr<Snapshot> &snapshot) {
    std::tm tm = {};
    std::istringstream iss(snapshot->getEventTime());
    iss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    int year = tm.tm_year + 1900;
    int month = tm.tm_mon + 1;
    int day = tm.tm_mday;

    std::ostringstream oss;
    oss << std::put_time(&tm, "%H-%M-%S");
    std::string timeStr = oss.str();

    ImageInfo dt;
    dt.year = year;
    dt.month = month;
    dt.day = day;
    dt.time = timeStr;
    dt.dayFolderPath = baseFolder / "snapshot";

    std::string baseName = snapshot->getCameraIp() + ".jpeg";
    fs::path fullPath = dt.dayFolderPath / baseName;

    dt.imagePath = fullPath;

    return dt;
}

