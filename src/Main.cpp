#include <thread>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <mutex>

#include "client/CameraClientLauncher.h"
#include "Config.h"
#include "SharedQueue.h"
#include "app/LPRecognizerService.h"
#include "app/DetectionService.h"
#include "package_sending/PackageSender.h"
#include "package_sending/Package.h"
#include "package_sending/Snapshot.h"
#include "package_sending/SnapshotSender.h"
#include "vehicle_detector/CarDetectionEngine.h"
#include "car_model/MobileNetV2.h"
#include "car_model/MobileNet.h"
#include "car_model/VehicleRecognizer.h"
#include "car_model/MMCClassifier.h"
#include "car_model/CarTypeClassifier.h"
#include "Template.h"

using namespace std;

atomic<bool> shutdownFlag = false;
condition_variable shutdownEvent;
mutex shutdownMutex;

void signalHandler(int signum) {
    cout << "signal is to shutdown" << endl;
    shutdownFlag = true;
    shutdownEvent.notify_all();
}

int main(int argc, char *argv[]) {
    if (IMSHOW) {
        char env[] = "DISPLAY=:1";
        putenv(env);
    }

    string configFileName, templateFileName;
    if (argc <= 1)
        configFileName = "config.json";
    else
        configFileName = argv[1];

    templateFileName = "template.json";

    if (!Config::parseJson(configFileName))
        return -1;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGKILL, signalHandler);
    signal(SIGHUP, signalHandler);
    signal(SIGABRT, signalHandler);
    auto lpRecognizer = make_unique<LPRecognizer>();
    auto packageQueue = make_shared < SharedQueue < shared_ptr < Package>>>();
    auto snapshotQueue = make_shared < SharedQueue < shared_ptr < Snapshot>>>();
    vector <shared_ptr<IThreadLauncher>> services;
    auto camerasVector = Config::getCameras();
    auto allCameras = Config::getAllCameras();
    vector < shared_ptr < SharedQueue < unique_ptr < FrameData >> >> frameQueues;

    auto vehicleDetectionEngine = make_shared<CarDetectionEngine>();
    auto detectionEngine = make_shared<DetectionEngine>();
    for (const auto &cameras: camerasVector) {
        std::unordered_map<std::string, Constants::CountryCode> templates;
        if (Template::parseJson(templateFileName)) {
            templates = Template::getTemplateMap();
        }
        auto lpRecognizerService = make_shared<LPRecognizerService>(packageQueue, cameras,
                                                                    Config::getRecognizerThreshold(),
                                                                    Config::getCalibrationSizes(),
                                                                    Config::getStatInterval(), templates);
        services.emplace_back(lpRecognizerService);
        for (const auto &camera: cameras) {
            auto frameQueue = make_shared < SharedQueue < unique_ptr < FrameData>>>();

            if (camera.whatTypeOfRecognitionEnabled() == Constants::VehicleClassificationType::noType) {
                auto detectionService = make_shared<DetectionService>(frameQueue, snapshotQueue,
                                                                      make_unique<CarDetection>(
                                                                              vehicleDetectionEngine->createExecutionContext()),
                                                                      make_unique<VehicleRecognizer>(),
                                                                      lpRecognizerService,
                                                                      make_shared<Detection>(
                                                                              detectionEngine->createExecutionContext(),
                                                                              Config::getDetectorThreshold()),
                                                                      camera, Config::getTimeSentSnapshots(),
                                                                      Config::getCalibrationSizes());
                services.emplace_back(detectionService);
            } else if (camera.whatTypeOfRecognitionEnabled() == Constants::VehicleClassificationType::carType ||
                       camera.whatTypeOfRecognitionEnabled() == Constants::VehicleClassificationType::baqordaType) {
                auto mobilenetEngineV2 = make_shared<MobileNetV2>(camera.whatTypeOfRecognitionEnabled());

                auto detectionService = make_shared<DetectionService>(frameQueue, snapshotQueue,
                                                                      make_unique<CarDetection>(
                                                                              vehicleDetectionEngine->createExecutionContext()),
                                                                      make_unique<CarTypeClassifier>(
                                                                              mobilenetEngineV2->createExecutionContext(),
                                                                              camera.whatTypeOfRecognitionEnabled(),
                                                                              camera.getDimensionCoords()),
                                                                      lpRecognizerService,
                                                                      make_shared<Detection>(
                                                                              detectionEngine->createExecutionContext(),
                                                                              Config::getDetectorThreshold()),
                                                                      camera, Config::getTimeSentSnapshots(),
                                                                      Config::getCalibrationSizes());
                services.emplace_back(detectionService);
            } else if (camera.whatTypeOfRecognitionEnabled() == Constants::VehicleClassificationType::modelType) {
                auto mobilenetEngine = make_shared<MobileNet>();
                auto detectionService = make_shared<DetectionService>(frameQueue, snapshotQueue,
                                                                      make_unique<CarDetection>(
                                                                              vehicleDetectionEngine->createExecutionContext()),
                                                                      make_unique<MMCClassifier>(
                                                                              mobilenetEngine->createExecutionContext(),
                                                                              camera.whatTypeOfRecognitionEnabled(),
                                                                              camera.getDimensionCoords()),
                                                                      lpRecognizerService,
                                                                      make_shared<Detection>(
                                                                              detectionEngine->createExecutionContext(),
                                                                              Config::getDetectorThreshold()),
                                                                      camera, Config::getTimeSentSnapshots(),
                                                                      Config::getCalibrationSizes());
                services.emplace_back(detectionService);
            }
            frameQueues.push_back(std::move(frameQueue));
        }
    }


    shared_ptr <IThreadLauncher> clientStarter;
    clientStarter = make_shared<CameraClientLauncher>(Config::getAllCameras(), frameQueues,
                                                      Config::useGPUDecode());
    services.emplace_back(clientStarter);

    auto packageSender = make_shared<PackageSender>(packageQueue, Config::getAllCameras(), Config::getBaseFolder(),
                                                    Config::useImageWriter());
    services.emplace_back(packageSender);


    auto snapshotSender = make_shared<SnapshotSender>(snapshotQueue, Config::getAllCameras(), Config::useImageWriter(),
                                                      Config::getBaseFolder());
    services.emplace_back(snapshotSender);

    vector <thread> threads;
    for (const auto &service: services) {
        threads.emplace_back(&IThreadLauncher::run, service);
    }

    unique_lock <mutex> shutdownLock(shutdownMutex);
    while (!shutdownFlag) {
        auto timeout = chrono::hours(24);
        if (shutdownEvent.wait_for(shutdownLock, timeout, [] { return shutdownFlag.load(); })) {
            cout << "shutting all services" << endl;
        }
    }

    for (int i = 0; i < services.size(); i++) {
        services[i]->shutdown();
        if (threads[i].joinable())
            threads[i].join();
    }
}
