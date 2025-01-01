#include "FrameData.h"

using namespace std;

FrameData::FrameData(string ip, cv::Mat frame, double rtpTimestamp,
                     chrono::high_resolution_clock::time_point startTime)
        : ip{move(ip)}, frame{move(frame)} {

    this->rtpTimestamp = rtpTimestamp;
    setStartTime(startTime);
}

const string &FrameData::getIp() {
    return ip;
}

const cv::Mat &FrameData::getFrame() {
    return frame;
}


double FrameData::getRTPtimestamp() const {
    return rtpTimestamp;
}
