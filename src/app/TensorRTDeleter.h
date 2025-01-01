//
// Created by kartykbayev on 6/2/22.
//

#pragma once

struct TensorRTDeleter {
    template<typename T>
    void operator()(T *obj) const {
        delete obj;
    }
};

template<typename T>
inline std::shared_ptr<T> infer_object(T *obj) {
    if (!obj) {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, TensorRTDeleter());
}