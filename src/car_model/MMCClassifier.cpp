//
// Created by artyk on 11/9/2023.
//

#include "MMCClassifier.h"

using namespace std;
using namespace nvinfer1;

MMCClassifier::MMCClassifier(nvinfer1::IExecutionContext *iExecutionContext,
                             Constants::VehicleClassificationType type, DimensionCoords dimensionCoords)
        : VehicleRecognizer() {

    imgWidth = Constants::CLS_TYPE_TO_IMG_WIDTH.find(type)->second;
    imgHeight = Constants::CLS_TYPE_TO_IMG_HEIGHT.find(type)->second;
    outputSize = Constants::CLS_TYPE_TO_OUTPUT_SIZE.find(type)->second;
    inputSize = imgWidth * imgHeight * IMG_CHANNELS;


    cudaMalloc(&cudaBuffer[0], inputSize * sizeof(float));
    cudaMalloc(&cudaBuffer[1], outputSize * sizeof(float));

    cudaStreamCreate(&stream);

    this->executionContext = iExecutionContext;
    COORDS = dimensionCoords;
}

MMCClassifier::~MMCClassifier() {
    cudaFree(cudaBuffer[0]);
    cudaFree(cudaBuffer[1]);

    cudaStreamDestroy(stream);

}


pair<string, double> MMCClassifier::classify(const std::shared_ptr<LicensePlate> &licensePlate) {
    cv::Mat cropped;

    auto flag = licensePlate->doesCarBBoxDefined() ? "model" : "custom";

    if (!licensePlate->doesCarBBoxDefined()) {
        cv::Mat frame = licensePlate->getCarImage();

        auto centerPoint = licensePlate->getCenter();
        int x_0 = centerPoint.x - COORDS.x1 < 0 ? 0 : centerPoint.x - COORDS.x1;
        int y_0 = centerPoint.y - COORDS.y1 < 0 ? 0 : centerPoint.y - COORDS.y1;
        int x_1 = centerPoint.x + COORDS.x2 > frame.cols ? frame.cols - 1 : centerPoint.x + COORDS.x2;
        int y_1 = centerPoint.y + COORDS.y2 > frame.rows ? frame.rows - 1 : centerPoint.y + COORDS.y2;


        cropped = frame(cv::Range(y_0, y_1), cv::Range(x_0, x_1));
    } else {
        cropped = licensePlate->getVehicleImage();
    }

    cv::Mat resized, converted;
    cv::resize(cropped, resized, cv::Size{imgWidth, imgHeight});
    resized.convertTo(converted, CV_32F);

    vector<float> out;
    out.resize(outputSize, 0.0);
    {
        cudaMemcpyAsync(cudaBuffer[0], converted.data, MAX_BATCH_SIZE * inputSize * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        executionContext->enqueue(MAX_BATCH_SIZE, cudaBuffer, stream, nullptr);


        cudaMemcpyAsync(out.data(), cudaBuffer[1], MAX_BATCH_SIZE * outputSize * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
    }

    vector<float> carModelOut;
    carModelOut.resize(CAR_MODEL_OUTPUT_SIZE, 0.0);

    for (int i = 0; i < CAR_MODEL_OUTPUT_SIZE; ++i) {
        carModelOut[i] = out[i];
    }

    vector<float> resultSoftmax = softmax(carModelOut);

    int carModelInd = 0;
    for (int i = 1; i < resultSoftmax.size(); i++) {
        if (resultSoftmax[i] > resultSoftmax[carModelInd]) carModelInd = i;
    }

    string car_type = Labels::CAR_MODEL_LABELS[carModelInd];
    float prob = resultSoftmax[carModelInd];

    if (prob >= CAR_MODEL_RECOGNITION_THRESHOLD)
        return make_pair(car_type, prob);
    else
        return make_pair("NotDefined", 0.0);


}

