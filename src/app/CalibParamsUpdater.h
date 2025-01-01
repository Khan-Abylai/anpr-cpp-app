#pragma once

#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "CalibParams.h"

class CalibParamsUpdater : public IThreadLauncher, public ILogger {
public:
    explicit CalibParamsUpdater(std::shared_ptr<CalibParams> calibParams);

    void run() override;

    void shutdown() override;

private:
    const int SLEEP_TIME_MINUTES = 5;

    std::condition_variable shutdownEvent;
    std::mutex updateMaskMutex;

    std::shared_ptr<CalibParams> calibParams;
    std::thread backgroundThread;

    void periodicallyUpdateMask();
};
