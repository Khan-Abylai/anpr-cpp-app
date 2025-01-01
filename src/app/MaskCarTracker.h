#pragma once

#include "BaseCarTracker.h"
#include "CalibParams.h"
#include "CalibParamsUpdater.h"

class MaskCarTracker : public BaseCarTracker {
public:
    explicit MaskCarTracker(std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                            const std::shared_ptr<CalibParams> &calibParams,
                            const CameraScope &cameraScope);

    void track(const std::shared_ptr<LicensePlate> &licensePlate) override;

    void run() override;

    void shutdown() override;

    void periodicallyCheckCurrentCarLifeTime();

private:
    bool useDirection;
    std::string originPoint;
    bool isForward;
    int platesCount;
    float timeBetweenResendingPlates;

    std::string cameraIp;
    std::shared_ptr<CalibParams> calibParams;
    std::unique_ptr<CalibParamsUpdater> calibParamsUpdater;

    bool isPlateAlreadySent{false};
    bool isLogGenerated{false};
    double lastTimeLPSent = 0;
    std::thread backgroundThread;
    const int CHECK_CURRENT_CAR_SECONDS = 1;

    std::condition_variable shutdownEvent;
    std::mutex updateMaskMutex;

    void sendMostCommonPlate() override;

    void saveFrame(const std::shared_ptr<LicensePlate> &licensePlate);

    bool isSufficientTimePassedToSendPlate();

    void considerToResendLP();

};
