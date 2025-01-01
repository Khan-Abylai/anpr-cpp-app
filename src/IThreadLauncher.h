#pragma once

#include <atomic>

class IThreadLauncher {
public:
    virtual void run() = 0;

    virtual void shutdown() = 0;

    virtual ~IThreadLauncher() = default;

protected:
    bool shutdownFlag = false;
};