#include "MaskCarTracker.h"

using namespace std;

MaskCarTracker::MaskCarTracker(shared_ptr<SharedQueue<shared_ptr<Package>>> packageQueue,
                               const shared_ptr<CalibParams> &calibParams, const CameraScope &cameraScope)
        : BaseCarTracker(std::move(packageQueue), cameraScope.getCameraIp()), calibParams{(calibParams)},
          platesCount{cameraScope.getPlatesCount()},
          timeBetweenResendingPlates{cameraScope.getTimeBetweenResendingPlates()},
          useDirection{cameraScope.isUseDirectionFlag()},
          vehicleClassificationType{cameraScope.whatTypeOfRecognitionEnabled()},
          originPoint(cameraScope.getOriginPoint()), isForward{cameraScope.isForwardFlagEnabled()},
          cameraIp(cameraScope.getCameraIp()) {
}

void MaskCarTracker::track(const shared_ptr<LicensePlate> &licensePlate) {
    licensePlate->setOverallTime();

    if (!currentCar || !currentCar->isLicensePlateBelongsToCar(licensePlate, lastFrameRTPTimestamp)) {

        if (currentCar && !isPlateAlreadySent) {
            if (!calibParams->isLicensePlateInSelectedArea(currentCar->getMostCommonLicensePlate(false, false),
                                                           "sub") && currentCar->doesPlatesCollected()) {
                LOG_INFO("plate %s was not sent -> it is not in the mask",
                         currentCar->getMostCommonLicensePlate(true, false)->getPlateLabel().data());
            } else if (!currentCar->doesPlatesCollected() &&
                       !calibParams->isLicensePlateInSelectedArea(currentCar->getMostCommonLicensePlate(false, false),
                                                                  "sub")) {
                LOG_INFO("plate %s was not sent -> plates not reached to the %d",
                         currentCar->getMostCommonLicensePlate(true, false)->getPlateLabel().data(), platesCount);
            } else {
                LOG_INFO("plate %s was not sent -> due to some other reasons",
                         currentCar->getMostCommonLicensePlate(true, false)->getPlateLabel().data());
            }
        }

        currentCar = createNewCar(platesCount, cameraIp, originPoint,
                                  cv::Point2f{(float)licensePlate->getCarImage().cols, (float)licensePlate->getCarImage().rows});
        isPlateAlreadySent = false;
        isLogGenerated = false;

        LOG_INFO("tracking new car %s", licensePlate->getPlateLabel().data());
    }

    lastFrameRTPTimestamp = licensePlate->getRTPtimestamp();
    currentCar->addTrackingPoint(licensePlate->getCenter(), isPlateAlreadySent);

    currentCar->addLicensePlateToCount(licensePlate);
    currentCar->setBestFrame();

    considerToResendLP();

    if (isPlateAlreadySent) return;

    if ((!useDirection || (((isForward && currentCar->getDirection() == Directions::forward) ||
                            (!isForward && currentCar->getDirection() == Directions::reverse)) && useDirection)) &&
        vehicleClassificationType == Constants::VehicleClassificationType::noType) {

        if (calibParams->isLicensePlateInSelectedArea(licensePlate, "sub") && currentCar->doesPlatesCollected()) {
            sendMostCommonPlate();
        }
    } else if (vehicleClassificationType != Constants::VehicleClassificationType::noType && !useDirection) {
        if (calibParams->isLicensePlateInSelectedArea(licensePlate, "main") &&
            !calibParams->isLicensePlateInSelectedArea(licensePlate, "sub")) {
            if ((string) licensePlate->getCarModel() != "NotDefined")
                currentCar->addTrackingCarModel(licensePlate->getCarModel(), licensePlate->getCarModelProb());
        } else if (calibParams->isLicensePlateInSelectedArea(licensePlate, "main") &&
                   calibParams->isLicensePlateInSelectedArea(licensePlate, "sub")) {
            if (currentCar->doesPlatesCollected() && currentCar->getCarModelWithOccurrence().second.first == 0 &&
                (string) licensePlate->getCarModel() != "NotDefined") {
                currentCar->addTrackingCarModel(licensePlate->getCarModel(), licensePlate->getCarModelProb());
                sendMostCommonPlate();
            } else if (currentCar->doesPlatesCollected() && currentCar->getCarModelWithOccurrence().second.first != 0) {
                if ((string) licensePlate->getCarModel() != "NotDefined")
                    currentCar->addTrackingCarModel(licensePlate->getCarModel(), licensePlate->getCarModelProb());
                sendMostCommonPlate();
            } else if (currentCar->doesPlatesDoubleCollected() &&
                       currentCar->getCarModelWithOccurrence().second.first == 0) {
                if ((string) licensePlate->getCarModel() != "NotDefine") {
                    currentCar->addTrackingCarModel(licensePlate->getCarModel(), licensePlate->getCarModelProb());
                } else {
                    currentCar->addTrackingCarModel("NotDefined", 0.0);
                }
                sendMostCommonPlate();
            }
        }
    }
}

void MaskCarTracker::considerToResendLP() {
    shared_ptr<LicensePlate> mostCommonLicensePlate = currentCar->getMostCommonLicensePlate(false, false);

    if (!useDirection && isPlateAlreadySent && isSufficientTimePassedToSendPlate() &&
        calibParams->isLicensePlateInSelectedArea(mostCommonLicensePlate, "sub")) {
        LOG_INFO("resending plate....");

        if (vehicleClassificationType != Constants::VehicleClassificationType::noType)
            mostCommonLicensePlate->setCarModel(currentCar->getCarModelWithOccurrence().first);
        else
            mostCommonLicensePlate->setCarModel("NotDefined");

        currentCar->addTrackingCarModel(mostCommonLicensePlate->getCarModel(), mostCommonLicensePlate->getCarModelProb());

        mostCommonLicensePlate->setDirection("forward");
        mostCommonLicensePlate->setRealTimeOfEvent(currentCar->getOverallTime());
        createAndPushPackage(mostCommonLicensePlate);
        if (vehicleClassificationType != Constants::VehicleClassificationType::noType)
            currentCar->mostCommonCarModelsShow();
        lastTimeLPSent = lastFrameRTPTimestamp;
    } else if ((useDirection && ((isForward && currentCar->getDirection() == Directions::forward) ||
                                 (!isForward && currentCar->getDirection() == Directions::reverse))) &&
               isPlateAlreadySent && isSufficientTimePassedToSendPlate() &&
               calibParams->isLicensePlateInSelectedArea(mostCommonLicensePlate, "sub")) {
        LOG_INFO("resending plate....");
        mostCommonLicensePlate->setCarModel("NotDefined");
        mostCommonLicensePlate->setDirection("forward");
        mostCommonLicensePlate->setRealTimeOfEvent(currentCar->getOverallTime());
        createAndPushPackage(mostCommonLicensePlate);
        lastTimeLPSent = lastFrameRTPTimestamp;
    }
}

bool MaskCarTracker::isSufficientTimePassedToSendPlate() {
    return lastFrameRTPTimestamp - lastTimeLPSent >= timeBetweenResendingPlates;
}

void MaskCarTracker::sendMostCommonPlate() {
    currentCar->setOverallTime();
    shared_ptr<LicensePlate> mostCommonLicensePlate = currentCar->getMostCommonLicensePlate();

    if (vehicleClassificationType != Constants::VehicleClassificationType::noType)
        mostCommonLicensePlate->setCarModel(currentCar->getCarModelWithOccurrence().first);
    else
        mostCommonLicensePlate->setCarModel("NotDefined");

    mostCommonLicensePlate->setDirection(("forward"));
    mostCommonLicensePlate->setRealTimeOfEvent(currentCar->getOverallTime());
    createAndPushPackage(mostCommonLicensePlate);

    isPlateAlreadySent = true;
    lastTimeLPSent = lastFrameRTPTimestamp;
}

void MaskCarTracker::run() {
    calibParamsUpdater = make_unique<CalibParamsUpdater>(calibParams);
    calibParamsUpdater->run();
    backgroundThread = thread(&MaskCarTracker::periodicallyCheckCurrentCarLifeTime, this);
}

void MaskCarTracker::shutdown() {
    LOG_INFO("service is shutting down");
    calibParamsUpdater->shutdown();
    shutdownFlag = true;
    shutdownEvent.notify_one();
    if (backgroundThread.joinable())
        backgroundThread.join();
}

void MaskCarTracker::saveFrame(const shared_ptr<LicensePlate> &licensePlate) {
    auto curPlate = currentCar->getMostCommonLicensePlate();
    cv::imwrite(Constants::IMAGE_DIRECTORY + licensePlate->getPlateLabel() + Constants::JPG_EXTENSION,
                licensePlate->getCarImage());
}

void MaskCarTracker::periodicallyCheckCurrentCarLifeTime() {
    unique_lock<mutex> lock(updateMaskMutex);
    auto timeout = chrono::seconds(CHECK_CURRENT_CAR_SECONDS);
    while (!shutdownFlag) {
        if (!shutdownEvent.wait_for(lock, timeout, [this] { return shutdownFlag; })) {

            if (currentCar && !isPlateAlreadySent && !isLogGenerated) {
                auto timeDiff = time(nullptr) - currentCar->getLifeTimeOfCar();

                if (timeDiff > 15) {
                    LOG_INFO("Current Car %s was not sent in 15 seconds",
                             currentCar->getMostCommonLicensePlate(false, false)->getPlateLabel().data());
                    isLogGenerated = true;
                }
            }
        }
    }
}

