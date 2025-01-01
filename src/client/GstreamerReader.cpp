#include "GstreamerReader.h"
#include "string"
#include <filesystem>

using namespace std;

GstreamerReader::GstreamerReader(const CameraScope &cameraScope, bool useGPUDecode,
                                 shared_ptr<SharedQueue<unique_ptr<FrameData>>> frameQueue)
        : ILogger("Camera " + cameraScope.getCameraIp()),
          cameraIp{cameraScope.getCameraIp()},
          useGPUDecode{useGPUDecode}, frameQueue{std::move(frameQueue)}, rtspUrl{cameraScope.getRtspUrl()} {
    LOG_INFO("RTSP URL: %s ", rtspUrl.c_str());
    gst_init(nullptr, nullptr);
    createStreamDecodingPipeline();
}

void GstreamerReader::createStreamDecodingPipeline() {
    string protocol = "protocols=tcp";
    string decode = (useGPUDecode) ? "nvdec ! gldownload" : "decodebin";
    auto streamingPipelineString = "rtspsrc " + protocol + " location=" + rtspUrl +
                                   " name=source latency=0 !  rtph264depay ! h264parse ! " + decode +
                                   " ! videoconvert ! video/x-raw, format=(string)BGR !  appsink name=sink emit-signals=true ";

//    string streamingPipelineString = "filesrc location=../videos/new_video_baqorda.mp4 ! decodebin !  videoconvert !  video/x-raw, format=(string)BGR ! appsink name=sink emit-signals=true ";
    decodingStreamPipeline = gst_parse_launch(streamingPipelineString.c_str(), nullptr);
    if (!GST_IS_PIPELINE(decodingStreamPipeline))
        LOG_ERROR("rtsp pipeline not initialized");

    auto sink = gst_bin_get_by_name(GST_BIN(decodingStreamPipeline), "sink");
    GstAppSinkCallbacks callbacks = {nullptr, nullptr, newDecodedSample};
    gst_app_sink_set_callbacks(GST_APP_SINK(sink), &callbacks, this, nullptr);
    gst_object_unref(sink);
}

const string &GstreamerReader::getCameraIp() const {
    return cameraIp;
}

void GstreamerReader::resetCameraState() {
    connected = false;
    frameNumber = 0;
}

void GstreamerReader::closePipeline() {
    if (decodingStreamPipeline) {
        gst_element_set_state(decodingStreamPipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(decodingStreamPipeline));
    }
}

void GstreamerReader::startPipeline() {
    gst_element_set_state(decodingStreamPipeline, GST_STATE_PLAYING);
}

void GstreamerReader::launchStream() {
    auto cameraConnectionWait = 1s;

    while (!shutdownFlag) {
        LOG_INFO("Launching camera");
        startPipeline();

        auto msg = gst_bus_timed_pop_filtered(decodingStreamPipeline->bus, GST_CLOCK_TIME_NONE,
                                              static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (msg) gst_message_unref(msg);

        closePipeline();
        createStreamDecodingPipeline();

        LOG_INFO("Disconnected");
        if (!connected.load()) {
            unique_lock<mutex> shutdownLock(shutdownMutex);

            if (shutdownEvent.wait_for(shutdownLock, cameraConnectionWait,
                                       [this] { return shutdownFlag.load(); }))
                break;

            if (cameraConnectionWait < CAMERA_CONNECTION_WAIT_MAX)
                cameraConnectionWait *= 2;
        } else cameraConnectionWait = 1s;

        resetCameraState();
    }
}

void GstreamerReader::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    frameQueue->push(nullptr);
    if (decodingStreamPipeline) {
        gst_element_set_state(decodingStreamPipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(decodingStreamPipeline));
    }
}

GstFlowReturn GstreamerReader::newDecodedSample(GstAppSink *appSink, gpointer data) {
    auto thisPointer = reinterpret_cast<GstreamerReader *>(data);
    GstSample *sample = gst_app_sink_pull_sample(appSink);

    if (thisPointer->frameNumber == 0) {
        thisPointer->LOG_INFO("Start time changed was %f now %ld %f", thisPointer->startTime, time(nullptr),
                              getRTPtimestamp(sample));
        thisPointer->startTime = time(nullptr) - getRTPtimestamp(sample);
    }
    thisPointer->frameNumber++;

    if (!thisPointer->connected.load()) {
        if (isKeyFrame(sample))
            thisPointer->connected = true;
        else {
            gst_sample_unref(sample);
            return GST_FLOW_OK;
        }
    }

    if (!sample) {
        g_print("Error sample");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *structure = gst_caps_get_structure(caps, 0);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    const int width = g_value_get_int(gst_structure_get_value(structure, "width"));
    const int height = g_value_get_int(gst_structure_get_value(structure, "height"));

    cv::Mat currentFrame(cv::Size(width, height), CV_8UC3, (char *) map.data);

    auto rtpTimestamp = getRTPtimestamp(buffer);
    auto difference = thisPointer->getTimeDiffBetweenFrames(rtpTimestamp);
    if (difference > thisPointer->DURATION_BETWEEN_FRAMES_TH)
        thisPointer->LOG_INFO("was delay: %lf", difference);

    auto startTime = chrono::high_resolution_clock::now();

    if (!currentFrame.empty()) {
        thisPointer->frameQueue->push(make_unique<FrameData>(thisPointer->cameraIp, std::move(currentFrame.clone()),
                                                             rtpTimestamp, startTime));
    }

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}

double GstreamerReader::getTimeDiffBetweenFrames(double rtpTimestamp) {
    auto difference = rtpTimestamp - lastRtpTimestamp;
    lastRtpTimestamp = rtpTimestamp;
    return difference;
}

double GstreamerReader::getRTPtimestamp(GstBuffer *buffer) {
    return static_cast<double>(GST_BUFFER_PTS (buffer)) / 1000000000;
}

double GstreamerReader::getRTPtimestamp(GstSample *sample) {
    return static_cast<double>(GST_BUFFER_PTS (gst_sample_get_buffer(sample))) / 1000000000;
}

bool GstreamerReader::isKeyFrame(GstSample *sample) {
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    return !GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
}