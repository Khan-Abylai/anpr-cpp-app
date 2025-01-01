#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ctime>
#include "ILogger.h"
#include <chrono>
#include <cstring>

using namespace std;

ILogger::ILogger() = default;

ILogger::ILogger(const string &prependMessage) {
    LOG_PREPEND_MESSAGE = prependMessage + ": ";
}

void ILogger::LOG_INFO(const string &format, ...) const {
    va_list args;
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%d-%m-%YT%H:%M:%S|", timeinfo);
    std::string str(buffer);

    auto newFormat = buffer + LOG_PREPEND_MESSAGE + format + "\n";
    va_start(args, format);
    vfprintf(stdout, newFormat.data(), args);
    fflush(stdout);
    va_end(args);
}

void ILogger::LOG_ERROR(const string &format, ...) const {
    auto newFormat = LOG_PREPEND_MESSAGE + format + "\n";
    va_list args;
    va_start(args, format);
    vfprintf(stderr, newFormat.data(), args);
    va_end(args);
}

void ILogger::LOG_DEBUG(const string &format, ...) const {
    va_list args;
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, sizeof(buffer), "%d-%m-%YT%H:%M:%S|", timeinfo);
    std::string str(buffer);

    auto newFormat = buffer + LOG_PREPEND_MESSAGE + format + "\n";
    va_start(args, format);
    vfprintf(stdout, newFormat.data(), args);
    fflush(stdout);
    va_end(args);
}
