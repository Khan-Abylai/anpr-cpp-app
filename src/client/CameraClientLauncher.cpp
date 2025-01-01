#include "CameraClientLauncher.h"

using namespace std;

CameraClientLauncher::CameraClientLauncher(const vector<CameraScope> &cameras,
                                           const vector<shared_ptr<SharedQueue<unique_ptr<FrameData>>>> &frameQueues,
                                           bool useGPUDecode)
        : ILogger("Camera Client Launcher ") {


    int index = 0;
    for (const auto &camera: cameras) {
        auto cameraReader = make_shared<GstreamerReader>(camera, useGPUDecode, frameQueues[index]);
        cameraReaders.push_back(std::move(cameraReader));
        index += 1;
    }
}

void CameraClientLauncher::run() {
    for (const auto &gstreamer: cameraReaders)
        threads.emplace_back(&GstreamerReader::launchStream, gstreamer);
}

void CameraClientLauncher::shutdown() {
    for (int i = 0; i < cameraReaders.size(); i++) {
        cameraReaders[i]->shutdown();
        if (threads[i].joinable())
            threads[i].join();
    }
}
