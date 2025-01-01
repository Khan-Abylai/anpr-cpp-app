#pragma once

#include <thread>

#include "../SharedQueue.h"
#include "../ILogger.h"
#include "../IThreadLauncher.h"
#include "GstreamerReader.h"
#include "../app/CameraScope.h"

class CameraClientLauncher : public IThreadLauncher, public ILogger {
public:
    CameraClientLauncher(const std::vector<CameraScope> &cameras,
                         const std::vector<std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>>> &frameQueues,
                         bool useGPUDecode);

    void run() override;

    void shutdown() override;

protected:
    std::vector<std::shared_ptr<GstreamerReader>> cameraReaders;
    std::vector<std::thread> threads;
};