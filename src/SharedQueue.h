#pragma once

#include <queue>
#include <mutex>
#include <exception>
#include <condition_variable>

template<typename T>

class SharedQueue {
    std::queue<T> queue_;
    mutable std::mutex m_;
    std::condition_variable dataCond;

public:

    SharedQueue &operator=(const SharedQueue &) = delete;

    SharedQueue(const SharedQueue &other) = delete;

    SharedQueue() = default;

    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(m_);
            queue_.push(std::move(item));
        }
        dataCond.notify_one();
    }

    T wait_and_pop() {
        std::unique_lock<std::mutex> lock(m_);
        while (queue_.empty()) {
            dataCond.wait(lock);
        }
        auto popped_item = std::move(queue_.front());
        queue_.pop();
        return std::move(popped_item);
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(m_);
        return queue_.empty();
    }

    unsigned size() const {
        std::lock_guard<std::mutex> lock(m_);
        return queue_.size();
    }

};