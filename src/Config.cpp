#include "Config.h"
#include "app/CameraScope.h"
#include "app/Constants.h"

using namespace std;
using json = nlohmann::json;

std::vector<std::vector<CameraScope>> cameras;
std::vector<CameraScope> allCameras;
bool useDecodeGpu = false;
bool useImageWriterService = false;
float timeBetweenSnapshotSend = 1.0f;

float recognizerThreshold = 0.9f;
float detectorThreshold = 0.8f;

float calibrationWidth = 1920.0f;
float calibrationHeight = 1080.0f;

int stat_interval = 1;
int lp_rec_service_count = 1;

std::string baseFolderPath;

bool Config::parseJson(const string &fileName) {
    try {
        ifstream configFile(fileName);

        if (!configFile.is_open())
            throw runtime_error("Config file not found");

        json configs = json::parse(configFile);

        if (configs.find("cameras") == configs.end())
            throw runtime_error("Camera IP Entities not defined");
        auto cameraEntities = configs["cameras"];

        if (configs.find("lp_rec_service_count") == configs.end())
            lp_rec_service_count = 1;
        else
            lp_rec_service_count = configs["lp_rec_service_count"].get<int>();

        for (const auto &cameraEntity: cameraEntities) {

            std::vector<string> keys = {"camera_ip", "rtsp_url", "result_send_endpoint", "node_ip",
                                        "snapshot_send_ip"};

            for (const auto &item: keys)
                if (cameraEntity.find(item) == cameraEntity.end())
                    throw runtime_error(item + " not defined");
            DimensionCoords dimensionCoords{};
            auto cameraIp = cameraEntity["camera_ip"].get<string>();
            auto rtspUrl = cameraEntity["rtsp_url"].get<string>();
            auto resultSendEndpoint = cameraEntity["result_send_endpoint"].get<string>();
            auto nodeIp = cameraEntity["node_ip"].get<string>();
            auto snapshotSendIp = cameraEntity["snapshot_send_ip"].get<string>();

            //////////////////////////////////////////////////////////////////////////////////////
            bool useSubMaskFlag, doesSecondaryNodeIpDefined, doesSecondarySnapshotIpDefined;
            bool isForwardFlag, useDirection;
            std::string secondaryResultSendIp, secondarySnapshotIp;
            int platesCount;
            float timeBetweenSendingPlates;
            //working with snapshots secondary snapshot send ip
            if (cameraEntity.find("secondary_snapshot_send_ip") == cameraEntity.end()) {
                doesSecondarySnapshotIpDefined = false;
            } else {
                doesSecondarySnapshotIpDefined = true;
                secondarySnapshotIp = cameraEntity["secondary_snapshot_send_ip"].get<string>();
            }
            //working with secondary result send endpoint
            if (cameraEntity.find("secondary_result_send_endpoint") == cameraEntity.end()) {
                doesSecondaryNodeIpDefined = false;
            } else {
                doesSecondaryNodeIpDefined = true;
                secondaryResultSendIp = cameraEntity["secondary_result_send_endpoint"].get<string>();
            }
            //working with secondary mask
            if (cameraEntity.find("use_mask_2") == cameraEntity.end() || cameraEntity["use_mask_2"].get<int>() == 1)
                useSubMaskFlag = true;
            else
                useSubMaskFlag = false;

            // working with the time between plate sending
            if (cameraEntity.find("time_between_sent_plates") == cameraEntity.end())
                timeBetweenSendingPlates = 5.0f;
            else
                timeBetweenSendingPlates = cameraEntity["time_between_sent_plates"].get<float>();

            // working with plates count
            if (cameraEntity.find("plates_count") == cameraEntity.end())
                platesCount = 5;
            else
                platesCount = cameraEntity["plates_count"].get<int>();


            if (cameraEntity.find("use_direction") == cameraEntity.end() ||
                cameraEntity["use_direction"].get<int>() == 0)
                useDirection = false;
            else
                useDirection = true;

            if ((cameraEntity.find("origin_point") == cameraEntity.end() &&
                 cameraEntity.find("is_forward") == cameraEntity.end() && useDirection) ||
                (cameraEntity.find("origin_point") == cameraEntity.end() &&
                 cameraEntity.find("is_forward") != cameraEntity.end() && useDirection) ||
                (cameraEntity.find("origin_point") != cameraEntity.end() &&
                 cameraEntity.find("is_forward") == cameraEntity.end() && useDirection))
                throw runtime_error("Use Direction enabled, but origin_point or is_forward not enabled");


            std::string originPoint;
            bool isForward;
            if (useDirection) {
                originPoint = cameraEntity["origin_point"].get<string>();
                isForward = cameraEntity["is_forward"].get<int>();

                if (originPoint != "left" && originPoint != "right" && originPoint != "center") {
                    throw std::runtime_error(
                            "Invalid originPoint value. Allowed values are 'left', 'right', and 'center'.");
                }
            } else {
                originPoint = "";
                isForward = false;
            }

            dimensionCoords.x1 = 250.0f;
            dimensionCoords.x2 = 250.0f;
            dimensionCoords.y1 = 500.0f;
            dimensionCoords.y2 = 150.0f;

            auto cameraScope = CameraScope(cameraIp, rtspUrl, resultSendEndpoint, nodeIp, snapshotSendIp,
                                           secondaryResultSendIp, doesSecondaryNodeIpDefined, secondarySnapshotIp,
                                           doesSecondarySnapshotIpDefined,
                                           timeBetweenSendingPlates,
                                           platesCount, useSubMaskFlag, useDirection, originPoint, isForward);
            allCameras.push_back(cameraScope);
        }

        cameras = reshapeVector(allCameras, lp_rec_service_count);


        if (configs.find("use_gpu_decode") == configs.end() || configs["use_gpu_decode"].get<int>() == 0)
            useDecodeGpu = false;
        else
            useDecodeGpu = true;

        if (configs.find("calibration_width") == configs.end())
            calibrationWidth = 1920.0;
        else
            calibrationWidth = configs["calibration_width"].get<float>();

        if (configs.find("calibration_height") == configs.end())
            calibrationHeight = 1080.0;
        else
            calibrationHeight = configs["calibration_height"].get<float>();

        if (configs.find("time_between_snapshot_send") == configs.end())
            timeBetweenSnapshotSend = 1.0f;
        else
            timeBetweenSnapshotSend = configs["time_between_snapshot_send"].get<float>();

        if (configs.find("rec_threshold") == configs.end())
            recognizerThreshold = 0.9f;
        else
            recognizerThreshold = configs["rec_threshold"].get<float>();

        if (configs.find("det_threshold") == configs.end())
            detectorThreshold = 0.8f;
        else
            detectorThreshold = configs["det_threshold"].get<float>();

        if (configs.find("base_folder") == configs.end())
            baseFolderPath = "/mnt/images";
        else
            baseFolderPath = configs["base_folder"].get<string>();

        if (configs.find("image_writer") == configs.end() || configs["image_writer"].get<int>() == 0)
            useImageWriterService = false;
        else
            useImageWriterService = true;

        if (configs.find("stat_interval") == configs.end())
            stat_interval = 1;
        else
            stat_interval = configs["stat_interval"].get<int>();

    } catch (exception &e) {
        cout << "Exception occurred during config parse: " << e.what() << endl;
        return false;
    }
    return true;
}

const std::vector<std::vector<CameraScope>> &Config::getCameras() {
    return cameras;
}

bool Config::useGPUDecode() {
    return useDecodeGpu;
}

const float &Config::getRecognizerThreshold() {
    return recognizerThreshold;
}

const float &Config::getDetectorThreshold() {
    return detectorThreshold;
}

const float &Config::getTimeSentSnapshots() {
    return timeBetweenSnapshotSend;
}

std::pair<float, float> Config::getCalibrationSizes() {
    return std::pair<float, float>{calibrationWidth, calibrationHeight};
}

const std::string &Config::getBaseFolder() {
    return baseFolderPath;
}

bool Config::useImageWriter() {
    return useImageWriterService;
}

int &Config::getStatInterval() {
    return stat_interval;
}

std::vector<std::vector<CameraScope>> Config::reshapeVector(const std::vector<CameraScope> &input, int n) {
    if (input.size() < n) {
        throw std::runtime_error("Camera count lower than lp_rec_service_count");
    }

    std::vector<std::vector<CameraScope>> output;

    int batchSize = (input.size() + n - 1) / n;

    output.resize(n);

    for (int i = 0; i < n; ++i) {
        int start = i * batchSize;
        int end = std::min(start + batchSize, static_cast<int>(input.size()));
        output[i].assign(input.begin() + start, input.begin() + end);
    }

    return output;
}

const std::vector<CameraScope> &Config::getAllCameras() {
    return allCameras;
}
























