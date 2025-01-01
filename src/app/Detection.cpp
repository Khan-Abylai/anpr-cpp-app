#include "Detection.h"

using namespace std;
using namespace nvinfer1;

Detection::Detection(IExecutionContext *executionContext,
                     const float &detectionThreshold) :
        ILogger("DetectorEngine"),
        LP_PROB_THRESHOLD{detectionThreshold} {

    this->executionContext = executionContext;

    cudaMalloc(&cudaBuffer[0], INPUT_SIZE * sizeof(float));
    cudaMalloc(&cudaBuffer[1], PLATE_OUTPUT_SIZE * sizeof(float));
    cudaStreamCreate(&stream);
}

Detection::~Detection() {
    cudaFree(cudaBuffer[0]);
    cudaFree(cudaBuffer[1]);
    cudaStreamDestroy(stream);
}

vector<tuple<float, shared_ptr<LicensePlate>>>
Detection::getLicensePlates(vector<float> lpPredictions, int frameWidth, int frameHeight) const {

    vector<tuple<float, shared_ptr<LicensePlate>>> licensePlatesWithProbs;

    float scaleWidth = static_cast<float>(frameWidth) / (float)IMG_WIDTH * (float)PLATE_GRID_SIZE;
    float scaleHeight = static_cast<float>(frameHeight) / (float)IMG_HEIGHT * (float)PLATE_GRID_SIZE;

    for (int row = 0; row < PLATE_GRID_HEIGHT; row++) {
        for (int col = 0; col < PLATE_GRID_WIDTH; col++) {

            float prob = 1 / (1 +
                              exp(-lpPredictions[12 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT +
                                                 col]));

            if (prob > LP_PROB_THRESHOLD) {

                float x = (1 / (1 + exp(-1 * lpPredictions[row * PLATE_GRID_HEIGHT + col])) + col) * scaleWidth;
                float y = (1 / (1 + exp(-1 *
                                        lpPredictions[PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT +
                                                      col])) + row) *
                          scaleHeight;

                float w = exp(lpPredictions[2 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col]) *
                          scaleWidth;
                float h = exp(lpPredictions[3 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col]) *
                          scaleHeight;

                float x1 = lpPredictions[4 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleWidth + x;
                float y1 = lpPredictions[5 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleHeight + y;

                float x2 = lpPredictions[6 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleWidth + x;
                float y2 = lpPredictions[7 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleHeight + y;

                float x3 = lpPredictions[8 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleWidth + x;
                float y3 = lpPredictions[9 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleHeight + y;

                float x4 = lpPredictions[10 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleWidth + x;
                float y4 = lpPredictions[11 * PLATE_GRID_WIDTH * PLATE_GRID_HEIGHT + row * PLATE_GRID_HEIGHT + col] *
                           scaleHeight + y;
                int new_h = static_cast<int>(((y2 - y1) + (y4 - y3)) / 2);
                int new_w = static_cast<int>(((x3 - x1) + (x4 - x2)) / 2);
                licensePlatesWithProbs.emplace_back(
                        prob, make_shared<LicensePlate>(static_cast<int>(x), static_cast<int>(y),
                                                                   abs(new_w), abs(new_h),
                                                                   static_cast<int>(floor(x1)),
                                                                   static_cast<int>(floor(y1)),
                                                                   static_cast<int>(floor(x2)),
                                                                   static_cast<int>(ceil(y2)),
                                                                   static_cast<int>(ceil(x3)),
                                                                   static_cast<int>(floor(y3)),
                                                                   static_cast<int>(ceil(x4)),
                                                                   static_cast<int>(ceil(y4)), prob));
            }
        }
    }

    return std::move(licensePlatesWithProbs);
}

vector<shared_ptr<LicensePlate>> Detection::detect(const cv::Mat &frame) {

    auto lpPredictions = executeEngine(frame);

    if (lpPredictions.empty()) return vector<shared_ptr<LicensePlate>>{};

    vector<tuple<float, shared_ptr<LicensePlate>>> licensePlatesWithProbs = getLicensePlates(std::move(lpPredictions),
                                                                                             frame.cols, frame.rows);
    sort(licensePlatesWithProbs.begin(), licensePlatesWithProbs.end(), greater<>());
    return nms(licensePlatesWithProbs);
}

vector<shared_ptr<LicensePlate>> Detection::nms(const vector<tuple<float, shared_ptr<LicensePlate>>> &licensePlates) const {

    vector<shared_ptr<LicensePlate>> filteredLicensePlates;
    vector<bool> isFiltered;
    isFiltered.reserve(licensePlates.size());
    for (int lpIndex = 0; lpIndex < licensePlates.size(); lpIndex++) {
        isFiltered[lpIndex] = false;
    }

    for (int lpIndex = 0; lpIndex < licensePlates.size(); lpIndex++) {
        if (isFiltered[lpIndex]) continue;

        isFiltered[lpIndex] = true;
        auto [_, licensePlate] = licensePlates[lpIndex];

        for (int filterLpIndex = lpIndex + 1; filterLpIndex < licensePlates.size(); filterLpIndex++) {
            auto &[_, anotherLicensePlate] = licensePlates[filterLpIndex];
            if (iou(licensePlate, anotherLicensePlate) > LP_NMS_THRESHOLD)
                isFiltered[filterLpIndex] = true;
        }

        filteredLicensePlates.emplace_back(std::move(licensePlate));
    }
    sort(filteredLicensePlates.begin(), filteredLicensePlates.end(),
         [](const shared_ptr<LicensePlate> &a, const shared_ptr<LicensePlate> &b) {
             return (a->getCenter().y == b->getCenter().y) ? (a->getCenter().x < b->getCenter().x)
                                                           : (a->getCenter().y < b->getCenter().y);
         });
    return std::move(filteredLicensePlates);
}


float Detection::iou(const shared_ptr<LicensePlate> &firstLp, const shared_ptr<LicensePlate> &secondLp) {
    float firstLpArea =
            (firstLp->getRightBottom().x - firstLp->getLeftTop().x + 1) *
            (firstLp->getRightBottom().y - firstLp->getLeftTop().y + 1);
    float secondLpArea =
            (secondLp->getRightBottom().x - secondLp->getLeftTop().x + 1) *
            (secondLp->getRightBottom().y - secondLp->getLeftTop().y + 1);

    float intersectionX2 = min(firstLp->getRightBottom().x, secondLp->getRightBottom().x);
    float intersectionY2 = min(firstLp->getRightBottom().y, secondLp->getRightBottom().y);
    float intersectionX1 = max(firstLp->getLeftTop().x, secondLp->getLeftTop().x);
    float intersectionY1 = max(firstLp->getLeftTop().y, secondLp->getLeftTop().y);

    float intersectionX = (intersectionX2 - intersectionX1 + 1);
    float intersectionY = (intersectionY2 - intersectionY1 + 1);

    if (intersectionX < 0)
        intersectionX = 0;

    if (intersectionY < 0)
        intersectionY = 0;

    float intersectionArea = intersectionX * intersectionY;

    return intersectionArea / (firstLpArea + secondLpArea - intersectionArea);
}


vector<float> Detection::executeEngine(const cv::Mat &frame) {

    auto flattenImage = prepareImage(frame);
    vector<float> lpPredictions;
    lpPredictions.resize(PLATE_OUTPUT_SIZE, 0);

    cudaMemcpyAsync(cudaBuffer[0], flattenImage.data(), MAX_BATCH_SIZE * INPUT_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    executionContext->enqueue(MAX_BATCH_SIZE, cudaBuffer, stream, nullptr);

    cudaMemcpyAsync(lpPredictions.data(), cudaBuffer[1], MAX_BATCH_SIZE * PLATE_OUTPUT_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    return std::move(lpPredictions);
}

std::vector<float> Detection::prepareImage(const cv::Mat &frame) const {
    vector<float> flattenedImage;
    flattenedImage.resize(INPUT_SIZE, 0);
    cv::Mat resizedFrame;
    resize(frame, resizedFrame, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    for (int row = 0; row < resizedFrame.rows; row++) {
        for (int col = 0; col < resizedFrame.cols; col++) {
            uchar *pixels = resizedFrame.data + resizedFrame.step[0] * row + resizedFrame.step[1] * col;
            flattenedImage[row * IMG_WIDTH + col] =
                    static_cast<float>(2 * ((float) pixels[0] / Constants::PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * ((float) pixels[1] / Constants::PIXEL_MAX_VALUE - 0.5));

            flattenedImage[row * IMG_WIDTH + col + 2 * IMG_HEIGHT * IMG_WIDTH] =
                    static_cast<float>(2 * ((float) pixels[2] / Constants::PIXEL_MAX_VALUE - 0.5));
        }
    }
    return std::move(flattenedImage);
}
