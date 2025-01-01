#pragma once

#include <future>
#include <string>
#include <ctime>
#include <shared_mutex>
#include <fstream>
#include "../app/CameraScope.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"

extern "C" {
#include <gst/gst.h>
#include <gst/app/app.h>
}

#include "../SharedQueue.h"
#include "FrameData.h"
#include "../IThreadLauncher.h"
#include "../ILogger.h"

class GstreamerReader : public ILogger {

public:

    GstreamerReader(const CameraScope &cameraScope, bool useGPUDecode,
                    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue);

    void launchStream();

    [[nodiscard]] const std::string &getCameraIp() const;

    void shutdown();

private:
    const double DURATION_BETWEEN_FRAMES_TH = 1.0;
    static constexpr std::chrono::seconds CAMERA_CONNECTION_WAIT_MAX = std::chrono::seconds(60);

    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue;
    std::string cameraIp;
    std::string rtspUrl;
    bool useGPUDecode = false;
    int frameNumber = 0;
    double lastRtpTimestamp = 0;
    double startTime = 0;

    GstElement *decodingStreamPipeline = nullptr;
    std::atomic<bool> connected = false;

    std::atomic<bool> shutdownFlag = false;
    std::condition_variable shutdownEvent;
    std::mutex shutdownMutex;

    static GstFlowReturn newDecodedSample(GstAppSink *appSink, gpointer data);

    void createStreamDecodingPipeline();

    void closePipeline();

    void startPipeline();

    static double getRTPtimestamp(GstBuffer *buffer);

    static double getRTPtimestamp(GstSample *sample);

    void resetCameraState();

    static bool isKeyFrame(GstSample *sample);

    double getTimeDiffBetweenFrames(double rtpTimestamp);
};
