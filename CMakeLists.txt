cmake_minimum_required(VERSION 3.15)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

project(parking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)

pkg_check_modules(ZMQ REQUIRED libzmq)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_RTP REQUIRED gstreamer-rtp-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories(${ZMQ_INCLUDE_DIRS})
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${CURL_INCLUDE_DIR})
include_directories(${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS} ${GST_RTP_INCLUDE_DIRS})

option(DEBUG_LOG "Debug Mode" OFF)
option(IMSHOW "Show Images" OFF)
option(IMSAVE "Save Images" OFF)
option(MORE_LOGS "Log tracking points" OFF)
option(LOG_FILE "Write logs into file" OFF)

if (DEBUG_LOG)
    add_definitions(-DDEBUG=1)
else ()
    add_definitions(-DDEBUG=0)
endif ()

if (IMSHOW)
    add_definitions(-DIMSHOW=1)
else ()
    add_definitions(-DIMSHOW=0)
endif ()

if (MORE_LOGS)
    add_definitions(-DMORE_LOGS=1)
else ()
    add_definitions(-DMORE_LOGS=0)
endif ()

if (IMSAVE)
    add_definitions(-DIMSAVE=1)
else ()
    add_definitions(-DIMSAVE=0)
endif ()

if (LOG_FILE)
    add_definitions(-DLOG_FILE=1)
else ()
    add_definitions(-DLOG_FILE=0)
endif ()

add_executable(parking
        src/SharedQueue.h
        src/Main.cpp
        src/client/FrameData.cpp
        src/client/FrameData.h
        src/app/DetectionEngine.cpp
        src/app/DetectionEngine.h
        src/app/Detection.cpp
        src/app/Detection.h
        src/app/Constants.h
        src/app/LicensePlate.cpp
        src/app/LicensePlate.h
        src/app/TensorRTEngine.cpp
        src/app/TensorRTEngine.h
        src/app/TensorRTDeleter.h
        src/app/TemplateMatching.cpp
        src/app/TemplateMatching.h
        src/app/Utils.cpp
        src/app/Utils.h
        src/package_sending/PackageSender.cpp
        src/package_sending/PackageSender.h
        src/IThreadLauncher.h
        src/ILogger.cpp
        src/ILogger.h
        src/app/LPRecognizerService.h
        src/app/LPRecognizerService.cpp
        src/package_sending/Package.h
        src/package_sending/Package.cpp
        src/Config.h
        src/Config.cpp
        src/client/GstreamerReader.cpp
        src/client/GstreamerReader.h
        src/app/LPRecognizer.cpp
        src/app/LPRecognizer.h
        src/ITimer.h
        src/RandomStringGenerator.h
        src/RandomStringGenerator.cpp
        src/app/DetectionService.cpp
        src/app/DetectionService.h
        src/app/Car.cpp
        src/app/Car.h
        src/app/Counter.h
        src/app/CalibParams.h
        src/app/CalibParams.cpp
        src/app/BaseCarTracker.cpp
        src/app/BaseCarTracker.h
        src/app/MaskCarTracker.cpp
        src/app/MaskCarTracker.h
        src/app/CalibParamsUpdater.h
        src/app/CalibParamsUpdater.cpp
        src/client/CameraClientLauncher.cpp
        src/client/CameraClientLauncher.h
        src/car_model/Labels.h
        src/Profile.h
        src/app/CameraScope.cpp
        src/app/CameraScope.h
        src/package_sending/SnapshotSender.cpp
        src/package_sending/SnapshotSender.h
        src/package_sending/Snapshot.cpp
        src/package_sending/Snapshot.h
        src/package_sending/ImageWriterService.cpp
        src/package_sending/ImageWriterService.h
        src/vehicle_detector/CarDetection.cpp
        src/vehicle_detector/CarDetection.h
        src/vehicle_detector/CarDetectionEngine.cpp
        src/vehicle_detector/CarDetectionEngine.h
        src/vehicle_detector/CarDimensions.cpp
        src/vehicle_detector/CarDimensions.h
        src/car_model/MobileNetV2.cpp
        src/car_model/MobileNetV2.h
        src/car_model/MobileNet.cpp
        src/car_model/MobileNet.h
        src/car_model/VehicleRecognizer.cpp
        src/car_model/VehicleRecognizer.h
        src/car_model/MMCClassifier.cpp
        src/car_model/MMCClassifier.h
        src/car_model/MMCClassifier.h
        src/car_model/CarTypeClassifier.cpp
        src/car_model/CarTypeClassifier.h
        src/Template.cpp
        src/Template.h
)

target_link_libraries(parking nvinfer ${OpenCV_LIBS} ${ZMQ_LIBRARIES} ${GST_LIBRARIES} ${GST_APP_LIBRARIES} ${GST_RTP_LIBRARIES}
        nlohmann_json::nlohmann_json /usr/local/cuda/lib64/libcudart.so
        ${CURL_LIBRARIES} /usr/local/lib/libcpr.so)