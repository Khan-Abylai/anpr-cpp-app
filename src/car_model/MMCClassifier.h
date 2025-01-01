//
// Created by artyk on 11/9/2023.
//

#include "VehicleRecognizer.h"
#include "Labels.h"

class MMCClassifier : public VehicleRecognizer {
public:
    MMCClassifier(nvinfer1::IExecutionContext *iExecutionContext,
                  Constants::VehicleClassificationType type, DimensionCoords dimensionCoords);

    ~MMCClassifier();

    std::pair<std::string, double> classify(const std::shared_ptr<LicensePlate> &licensePlate) override;
private:
    const int CAR_MODEL_OUTPUT_SIZE = 417;
    DimensionCoords COORDS{};
    int imgWidth;
    int imgHeight;
    void *cudaBuffer[2];
    nvinfer1::IExecutionContext *executionContext;
    int outputSize;
    const int
            MAX_BATCH_SIZE = Constants::CAR_MODEL_BATCH_SIZE;
    int inputSize{};
    cudaStream_t stream{};

};
