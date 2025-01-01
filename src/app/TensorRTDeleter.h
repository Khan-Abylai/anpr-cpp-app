#pragma once

struct TensorRTDeleter {
    template<typename T>
    void operator()(T *obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};