#pragma once

#include <iostream>
#include "NvInfer.h"

class TrtLogger : public nvinfer1::ILogger {
public:

    TrtLogger() : TrtLogger(Severity::kWARNING) {}

    TrtLogger(Severity severity) : reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) override {
        if (severity > reportableSeverity) return;

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kINFO};
};
