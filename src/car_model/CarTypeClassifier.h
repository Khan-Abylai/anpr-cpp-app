//
// Created by artyk on 11/9/2023.
//


#include "VehicleRecognizer.h"
#include "Labels.h"
class CarTypeClassifier: public VehicleRecognizer{
public:
    CarTypeClassifier(nvinfer1::IExecutionContext *iExecutionContext,
                      Constants::VehicleClassificationType type, DimensionCoords dimensionCoords);
    ~CarTypeClassifier();

    std::pair<std::string, double> classify(const std::shared_ptr<LicensePlate> &licensePlate) override;
private:
    DimensionCoords COORDS{};
    int imgWidth;
    int imgHeight;
    void *cudaBuffer[2]{};
    nvinfer1::IExecutionContext *executionContext;
    int outputSize;
    const int
            MAX_BATCH_SIZE = Constants::CAR_MODEL_BATCH_SIZE;
    int inputSize;
    cudaStream_t stream{};


};


