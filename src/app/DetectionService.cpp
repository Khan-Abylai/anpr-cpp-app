#include "DetectionService.h"

using namespace std;

DetectionService::DetectionService(std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue,
                                   const shared_ptr<SharedQueue<std::shared_ptr<Snapshot>>> &snapshotQueue,
                                   std::unique_ptr<CarDetection> vehicleDetection,
                                   std::unique_ptr<VehicleRecognizer> vehicleRecognizer,
                                   std::shared_ptr<LPRecognizerService> lpRecognizerService,
                                   std::shared_ptr<Detection> detection, const CameraScope &cameraScope,
                                   float timeBetweenSendingSnapshots, std::pair<float, float> calibrationSizes)
        : ILogger("DetectionService"), frameQueue{std::move(frameQueue)}, snapshotQueue{snapshotQueue},
          detection{std::move(detection)}, USE_MASK_2{cameraScope.isUseMask2Flag()},
          timeBetweenSendingSnapshots{timeBetweenSendingSnapshots}, lpRecognizerService{std::move(lpRecognizerService)},
          carDetection{std::move(vehicleDetection)}, vehicleRecognizer{std::move(vehicleRecognizer)},
          CALIBRATION_SIZES{calibrationSizes} {

    if (USE_MASK_2 || cameraScope.whatTypeOfRecognitionEnabled() != Constants::VehicleClassificationType::noType) {
        calibParams2 = make_shared<CalibParams>(cameraScope.getNodeIp(), cameraScope.getCameraIp(), calibrationSizes);
        calibParamsUpdater2 = make_unique<CalibParamsUpdater>(calibParams2);
        calibParamsUpdater2->run();
        calibParams2->getMask();
    }

    lpRecognizerService = std::move(lpRecognizerService);

}

shared_ptr<LicensePlate> DetectionService::getMaxAreaPlate(vector<shared_ptr<LicensePlate>> &licensePlates) {
    float maxArea = -1.0;
    shared_ptr<LicensePlate> licensePlate;
    for (auto &curPlate: licensePlates) {
        float area = curPlate->getArea();
        if (area > maxArea) {
            maxArea = area;
            licensePlate = std::move(curPlate);
        }
    }
    return licensePlate;
}

shared_ptr<LicensePlate> DetectionService::chooseOneLicensePlate(vector<shared_ptr<LicensePlate>> &licensePlates) {

    shared_ptr<LicensePlate> licensePlate;

    if (licensePlates.size() > 1)
        licensePlate = getMaxAreaPlate(licensePlates);
    else
        licensePlate = std::move(licensePlates.front());

    return licensePlate;
};

void DetectionService::run() {
    while (!shutdownFlag) {
        auto frameData = frameQueue->wait_and_pop();
        if (frameData == nullptr) continue;
        auto frame = frameData->getFrame();
        auto cameraScope = lpRecognizerService->getCameraScope(frameData->getIp());
        lastFrameRTPTimestamp = time(nullptr);
        if (lastFrameRTPTimestamp - lastTimeSnapshotSent >= timeBetweenSendingSnapshots &&
            !cameraScope.getSnapshotSendIp().empty()) {
            cv::Mat resizedFrame;
            cv::resize(frame.clone(), resizedFrame,
                       cv::Size{(int) CALIBRATION_SIZES.first, (int) CALIBRATION_SIZES.second});
            auto snapshot = make_shared<Snapshot>(frameData->getIp(), resizedFrame, cameraScope.getSnapshotSendIp(),
                                                  cameraScope.isUseSecondarySnapshotEndpoint(),
                                                  cameraScope.getOtherSnapshotSendEndpoint());
            snapshotQueue->push(std::move(snapshot));
            lastTimeSnapshotSent = time(nullptr);
        }

        if (lastFrameRTPTimestamp - lastTimeMonitoringDone >= timeBetweenMonitoringDone) {
            cv::Mat grayscale;
            cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
            cv::Scalar m = cv::mean(grayscale);
            auto val = m.val[0];
            LOG_DEBUG("Gray | Camera IP:%s grayscale is: %d", frameData->getIp().c_str(), int(val));
            lastTimeMonitoringDone = time(nullptr);
        }


        auto startTime = chrono::high_resolution_clock::now();
        auto detectionResult = detection->detect(frame);
        auto endTime = chrono::high_resolution_clock::now();

        if (detectionResult.empty()) continue;

//        pid_t pid = getpid(); // Получаем идентификатор текущего процесса
//        int result = kill(pid, SIGTERM);

        auto licensePlate = chooseOneLicensePlate(detectionResult);

        licensePlate->setDetectionTime(
                (double) chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count());
        licensePlate->setStartTime(frameData->getStartTime());

        licensePlate->setPlateImage(frame);
        licensePlate->setCameraIp(frameData->getIp());


        if (lpRecognizerService->getCameraScope(frameData->getIp()).whatTypeOfRecognitionEnabled() !=
            Constants::VehicleClassificationType::noType) {
            auto result = carDetection->carDetect(frame);
            if (!result.empty()) {
                auto centralPoint = licensePlate->getCenter();
                bool flag = false;
                for (const auto &res: result) {
                    cv::Rect rRect(res->getTopLeft(), res->getBottomRight());
                    rRect.x = max(rRect.x, 0);
                    rRect.y = max(rRect.y, 0);
                    rRect.width = min(frame.cols - 1 - rRect.x, rRect.width - 1);
                    rRect.height = min(frame.rows - 1 - rRect.y, rRect.height - 1);
                    if (rRect.contains(centralPoint) && centralPoint.y >= res->getCenter().y) {
                        flag = true;
                        cv::Mat vehicleImage = frame(rRect);
                        licensePlate->setVehicleImage(std::move(vehicleImage));
                        licensePlate->setCarBBoxFlag(true);
                    }
                }
                if (!flag)licensePlate->setCarBBoxFlag(false);
                result.clear();
            } else {
                licensePlate->setCarBBoxFlag(false);
            }
        } else {
            licensePlate->setCarBBoxFlag(false);
        }

        licensePlate->setCarImage(std::move(frame));
        licensePlate->setRTPtimestamp(frameData->getRTPtimestamp());

        if (USE_MASK_2 || lpRecognizerService->getCameraScope(frameData->getIp()).whatTypeOfRecognitionEnabled() !=
                          Constants::VehicleClassificationType::noType)
            if (!calibParams2->isLicensePlateInSelectedArea(licensePlate, "main")) {
                continue;
            }


        licensePlate->setResultSendUrl(cameraScope.getResultSendIp());
        if (cameraScope.isUseSecondaryEndpoint()) {
            licensePlate->setSecondaryUrlEnabledFlag(true);
            licensePlate->setSecondaryResultSendUrl(cameraScope.getOtherResultSendEndpoint());
        } else {
            licensePlate->setSecondaryUrlEnabledFlag(false);
        }


        if (lpRecognizerService->getCameraScope(frameData->getIp()).whatTypeOfRecognitionEnabled() !=
            Constants::VehicleClassificationType::noType) {
            auto [carModel, prob] = vehicleRecognizer->classify(licensePlate);
            licensePlate->setCarModel(carModel);
            licensePlate->setCarModelProb(prob);
        } else {
            licensePlate->setCarModel("NotDefined");
            licensePlate->setCarModelProb(0.0);
        }
        auto stop = 1;
        lpRecognizerService->addToQueue(std::move(licensePlate));
    }
}


void DetectionService::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    frameQueue->push(nullptr);
}

bool DetectionService::isChooseThisFrame() {
    srand(time(nullptr));
    auto randomNumber = 1 + rand() % 100; // generating number between 1 and 100
    return (randomNumber < 20);
}


void DetectionService::saveFrame(const shared_ptr<LicensePlate> &plate) {
    if (!isChooseThisFrame()) return;
    string fileName = RandomStringGenerator::generate(30, Constants::IMAGE_DIRECTORY, Constants::JPG_EXTENSION);
    auto frame = plate->getCarImage();
    cv::imwrite(fileName, frame);
}

