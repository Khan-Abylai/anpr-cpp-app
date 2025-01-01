#pragma once

#include <chrono>

#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "../client/FrameData.h"

#include "Detection.h"
#include "DetectionEngine.h"
#include "LPRecognizerService.h"
#include "../RandomStringGenerator.h"
#include "../package_sending/Snapshot.h"
#include "../vehicle_detector/CarDetection.h"
#include "../car_model/VehicleRecognizer.h"

class DetectionService : public IThreadLauncher, public ILogger {
public:
    DetectionService(std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue,
                     const std::shared_ptr<SharedQueue<std::shared_ptr<Snapshot>>> &snapshotQueue,
                     std::unique_ptr<CarDetection> vehicleDetection,
                     std::unique_ptr<VehicleRecognizer> vehicleRecognizer,
                     std::shared_ptr<LPRecognizerService> lpRecognizerService,
                     std::shared_ptr<Detection> detection,
                     const CameraScope &cameraScope,
                     float timeBetweenSendingSnapshots,
                     std::pair<float, float> calibrationSizes);

    void run() override;

    void shutdown() override;

private:
    static bool isChooseThisFrame();

    bool USE_MASK_2;
    std::pair<float, float> CALIBRATION_SIZES;
    time_t lastFrameRTPTimestamp = time(nullptr);
    time_t lastTimeSnapshotSent = time(nullptr);
    time_t lastTimeMonitoringDone = time(nullptr);
    float timeBetweenSendingSnapshots = 1.0;
    int timeBetweenMonitoringDone = 60;


    std::shared_ptr<Detection> detection;
    std::shared_ptr<LPRecognizerService> lpRecognizerService;
    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue;
    std::shared_ptr<SharedQueue<std::shared_ptr<Snapshot>>> snapshotQueue;

    static void saveFrame(const std::shared_ptr<LicensePlate> &plate);

    static std::shared_ptr<LicensePlate> getMaxAreaPlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates);


    std::shared_ptr<CalibParams> calibParams2;
    std::unique_ptr<CalibParamsUpdater> calibParamsUpdater2;

    static std::shared_ptr<LicensePlate>
    chooseOneLicensePlate(std::vector<std::shared_ptr<LicensePlate>> &licensePlates);

    std::unique_ptr<CarDetection> carDetection;
    std::unique_ptr<VehicleRecognizer> vehicleRecognizer;

};


