//
// Created by kartykbayev on 4/13/23.
//

#include "CarDetection.h"

using namespace std;
using namespace nvinfer1;

CarDetection::CarDetection(nvinfer1::IExecutionContext *executionContext) {
    this->executionContext = executionContext;

    cudaMalloc(&cudaBuffer[0], INPUT_SIZE * sizeof(float));
    cudaMalloc(&cudaBuffer[1], CAR_OUTPUT_SIZE * sizeof(float));
    cudaStreamCreate(&stream);
}

CarDetection::~CarDetection() {
    cudaFree(cudaBuffer[0]);
    cudaFree(cudaBuffer[1]);
    cudaStreamDestroy(stream);
}

std::vector<std::shared_ptr<CarDimensions>> CarDetection::carDetect(const cv::Mat &frame) {
    auto carPredictions = executeEngine(frame);
    auto carDimensions = getCarDimensions(std::move(carPredictions), frame.cols, frame.rows);
    return std::move(carDimensions);
}

std::vector<float> CarDetection::executeEngine(const cv::Mat &frame) {
    auto flattenImage = prepareImage(frame);
    vector<float> carPredictions;
    carPredictions.resize(CAR_OUTPUT_SIZE,0.0);

    cudaMemcpyAsync(cudaBuffer[0], flattenImage.data(), MAX_BATCH_SIZE * INPUT_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    executionContext->enqueue(MAX_BATCH_SIZE, cudaBuffer, stream, nullptr);

    cudaMemcpyAsync(carPredictions.data(), cudaBuffer[1], MAX_BATCH_SIZE * CAR_OUTPUT_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    return move(carPredictions);
}

std::vector<float> CarDetection::prepareImage(const cv::Mat &frame) {
    vector<float> flattenedImage;
    flattenedImage.resize(INPUT_SIZE, 0.0);
    cv::Mat resizedFrame;
    resize(frame, resizedFrame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    for (int row = 0; row < resizedFrame.rows; row++) {
        for (int col = 0; col < resizedFrame.cols; col++) {
            uchar *pixels = resizedFrame.data + resizedFrame.step[0] * row + resizedFrame.step[1] * col;
            flattenedImage[row * IMG_WIDTH + col] =
                    static_cast<float>(2 * (pixels[0] / PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * (pixels[1] / PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + 2 * IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * (pixels[2] / PIXEL_MAX_VALUE - 0.5));
        }
    }
    return move(flattenedImage);
}

std::vector<std::shared_ptr<CarDimensions>>
CarDetection::nms(const vector<std::tuple<float, std::shared_ptr<CarDimensions>>> &carDimensions) {
    vector<shared_ptr<CarDimensions>> filteredCarDimensions;
    vector<bool> isFiltered;
    isFiltered.resize(carDimensions.size(), false);
    for (int carDimensionIndex = 0; carDimensionIndex < carDimensions.size(); carDimensionIndex++) {
        isFiltered[carDimensionIndex] = false;
    }
    for (int carDimensionIndex = 0; carDimensionIndex < carDimensions.size(); carDimensionIndex++) {
        if (isFiltered[carDimensionIndex]) {
            continue;
        }

        isFiltered[carDimensionIndex] = true;
        auto [_, carDimension] = carDimensions[carDimensionIndex];

        for (int filterCarDimensionIndex = carDimensionIndex + 1;
             filterCarDimensionIndex < carDimensions.size(); filterCarDimensionIndex++) {
            auto &[_, anotherCarDimension] = carDimensions[filterCarDimensionIndex];
            auto iou_ = iou(carDimension, anotherCarDimension);
            if ( iou_> CAR_NMS_THRESHOLD) {
                isFiltered[filterCarDimensionIndex] = true;
            }
        }
        filteredCarDimensions.emplace_back(move(carDimension));
    }
    sort(filteredCarDimensions.begin(), filteredCarDimensions.end(),
         [](const shared_ptr<CarDimensions> &a, const shared_ptr<CarDimensions> &b) {
             return (a->getCenter().y == b->getCenter().y) ? (a->getCenter().x < b->getCenter().x) : (a->getCenter().y <
                                                                                                      b->getCenter().y);
         });
    return move(filteredCarDimensions);
}

float CarDetection::iou(const shared_ptr<CarDimensions> &firstCarDimension,
                        const shared_ptr<CarDimensions> &secondCarDimension) {
    float firstCarArea = firstCarDimension->getArea();
    float secondCarArea = secondCarDimension->getArea();

    float intersectionX2 = min(firstCarDimension->getCenter().x + firstCarDimension->getWidth() / 2,
                               secondCarDimension->getCenter().x + secondCarDimension->getWidth() / 2);

    float intersectionY2 = min(firstCarDimension->getCenter().y + firstCarDimension->getHeight() / 2,
                               secondCarDimension->getCenter().y + secondCarDimension->getHeight() / 2);

    float intersectionX1 = max(firstCarDimension->getCenter().x - firstCarDimension->getWidth() / 2,
                               secondCarDimension->getCenter().x - secondCarDimension->getWidth() / 2);

    float intersectionY1 = max(firstCarDimension->getCenter().y - firstCarDimension->getHeight() / 2,
                               secondCarDimension->getCenter().y - secondCarDimension->getHeight() / 2);

    float intersectionX = (intersectionX2 - intersectionX1 + 1);
    float intersectionY = (intersectionY2 - intersectionY1 + 1);

    if (intersectionX < 0) {
        intersectionX = 0;
    }

    if (intersectionY < 0) {
        intersectionY = 0;
    }

    float intersectionArea = intersectionX * intersectionY;

    return intersectionArea / (firstCarArea + secondCarArea - intersectionArea);
}

std::vector<std::shared_ptr<CarDimensions>>
CarDetection::getCarDimensions(std::vector<float> carPredictions, int frameWidth, int frameHeight) {
    vector<tuple<float, shared_ptr<CarDimensions>>> carDimensionsWithProb;

    float scaleWidth = static_cast<float>(frameWidth) / IMG_WIDTH * CAR_GRID_SIZE;
    float scaleHeight = static_cast<float>(frameHeight) / IMG_HEIGHT * CAR_GRID_SIZE;

    for (int row = 0; row < CAR_GRID_HEIGHT; row++) {
        for (int col = 0; col < CAR_GRID_WIDTH; col++) {

            float prob =
                    1 / (1 + exp(-carPredictions[4 * CAR_GRID_WIDTH * CAR_GRID_HEIGHT + row * CAR_GRID_HEIGHT + col]));
            if (prob > CAR_PROB_THRESHOLD) {

                float x = (1 / (1 + exp(-1 * carPredictions[row * CAR_GRID_HEIGHT + col])) + col) * scaleWidth;
                float y = (1 / (1 + exp(-1 * carPredictions[CAR_GRID_WIDTH * CAR_GRID_HEIGHT + row * CAR_GRID_HEIGHT +
                                                            col])) + row) * scaleHeight;

                float w = exp(carPredictions[2 * CAR_GRID_WIDTH * CAR_GRID_HEIGHT + row * CAR_GRID_HEIGHT + col]) *
                          scaleWidth;
                float h = exp(carPredictions[3 * CAR_GRID_WIDTH * CAR_GRID_HEIGHT + row * CAR_GRID_HEIGHT + col]) *
                          scaleHeight;

                carDimensionsWithProb.emplace_back(prob, make_shared<CarDimensions>(static_cast<int>(x),
                                                                                               static_cast<int>(y),
                                                                                               static_cast<int>(w),
                                                                                               static_cast<int>(h), frameWidth, frameHeight));
            }
        }
    }
    sort(carDimensionsWithProb.begin(), carDimensionsWithProb.end());
    return std::move(nms(carDimensionsWithProb));
}
