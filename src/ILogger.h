#pragma once

#include <string>
#include <memory>

class ILogger {
public:
    ILogger();

    explicit ILogger(const std::string& prependMessage);

    ~ILogger() = default;

protected:
    void LOG_INFO(const std::string &format, ...) const;

    void LOG_ERROR(const std::string &format, ...) const;

    void LOG_DEBUG(const std::string &format, ...) const;

private:
    std::string LOG_PREPEND_MESSAGE;

    FILE* infoLogFile{};
    FILE* errorLogFile{};
    FILE* debugLogFile{};
};