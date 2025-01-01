#pragma once

#include <chrono>

class ITimer {
public:
    void setStartTime(std::chrono::high_resolution_clock::time_point timestamp) {
        startTime = timestamp;
    }

    void setDetectionTime(double timestamp) {
        detectionTime = timestamp;
    }

    void setRecognizerTime(double timestamp) {
        recognizerTime = timestamp;
    }

    double getOverallTime() const {
        return overallTime;
    }

    void setOverallTime() {
        auto endTime = std::chrono::high_resolution_clock::now();
        this->overallTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    }

    void setOverallTime(double timestamp) {
        overallTime = timestamp;
    }

    std::chrono::high_resolution_clock::time_point getStartTime() {
        return startTime;
    }

    double getDetectionTime() const {
        return detectionTime;
    }

    double getRecognizerTime() const {
        return recognizerTime;
    }

private:
    double detectionTime;
    double recognizerTime;
    double overallTime;
    std::chrono::high_resolution_clock::time_point startTime;
};
