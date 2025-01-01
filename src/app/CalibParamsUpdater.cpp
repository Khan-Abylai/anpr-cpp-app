#include "CalibParamsUpdater.h"

using namespace std;

void CalibParamsUpdater::periodicallyUpdateMask() {
    unique_lock<mutex> lock(updateMaskMutex);
    auto timeout = chrono::minutes(SLEEP_TIME_MINUTES);
    while (!shutdownFlag) {
        if (!shutdownEvent.wait_for(lock, timeout, [this] { return shutdownFlag; })) {
            LOG_INFO("updating mask for %s.....", calibParams->getCameraIp().c_str());
            calibParams->getMask();
        }
    }
}

CalibParamsUpdater::CalibParamsUpdater(shared_ptr<CalibParams> calibParams) : ILogger(
        "Calib Params Updater " + calibParams->getCameraIp()), calibParams{std::move(calibParams)} {};

void CalibParamsUpdater::run() {
    backgroundThread = thread(&CalibParamsUpdater::periodicallyUpdateMask, this);
}

void CalibParamsUpdater::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    shutdownEvent.notify_one();
    if (backgroundThread.joinable())
        backgroundThread.join();
}
