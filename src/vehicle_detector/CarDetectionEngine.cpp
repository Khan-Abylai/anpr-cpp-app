//
// Created by kartykbayev on 4/13/23.
//

#include "CarDetectionEngine.h"

using namespace std;
using namespace nvinfer1;

CarDetectionEngine::CarDetectionEngine() {

    if (!filesystem::exists(ENGINE_NAME)) {
        createEngine();
        TensorRTEngine::serializeEngine(engine, ENGINE_NAME);
    }

    engine = TensorRTEngine::readEngine(ENGINE_NAME);
    if (!engine) {
        filesystem::remove(ENGINE_NAME);
        throw "Corrupted Engine";
    }
}

void CarDetectionEngine::createEngine() {
    TrtLogger trt_logger;
    auto builder = unique_ptr<IBuilder, TensorRTDeleter>(createInferBuilder(trt_logger), TensorRTDeleter());

    vector<float> weights;
    ifstream weightFile(Constants::modelWeightsFolder + WEIGHTS_FILENAME, ios::binary);

    float parameter;

    weights.reserve(weightFile.tellg() / 4); // char to float

    while (weightFile.read(reinterpret_cast<char *>(&parameter), sizeof(float))) {
        weights.push_back(parameter);
    }

    auto network = unique_ptr<INetworkDefinition, TensorRTDeleter>(builder->createNetworkV2(0), TensorRTDeleter());

    ITensor *inputLayer = network->addInput(NETWORK_INPUT_NAME.c_str(), DataType::kFLOAT,
                                            Dims3{IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH});

    ITensor *prevLayer;
    vector<int> channels{16, 32, 64, 128, 256, 512, 1024};


    DimsHW kernelSize{3, 3};
    int index = 0;
    prevLayer = inputLayer;

    vector<ITensor *> prevLayers;
    int convBlockSize = 2;

    for (int channelIndex = 0; channelIndex < channels.size(); channelIndex++) {
        for (int convBlockIndex = 0; convBlockIndex < convBlockSize; convBlockIndex++) {

            int convWeightsCount =
                    prevLayer->getDimensions().d[0] * channels[channelIndex] * kernelSize.d[0] * kernelSize.d[1];
            Weights convWeights{DataType::kFLOAT, &weights[index], convWeightsCount};
            index += convWeightsCount;

            int convBiasesCount = channels[channelIndex];
            Weights convBias{DataType::kFLOAT, &weights[index], convBiasesCount};
            index += convBiasesCount;

            auto convLayer = network->addConvolution(*prevLayer, channels[channelIndex], kernelSize, convWeights,
                                                     convBias);
            convLayer->setStride(DimsHW{1, 1});
            convLayer->setPadding(DimsHW{1, 1});

            for (int channel = 0; channel < channels[channelIndex]; channel++) {

                weights[index + channel] /= sqrt(weights[index + channels[channelIndex] * 3 + channel] + 1e-5);

                weights[index + channels[channelIndex] + channel] -=
                        weights[index + channels[channelIndex] * 2 + channel] * weights[index + channel];

                weights[index + channels[channelIndex] * 2 + channel] = 1.0;
            }

            Weights layerScale{DataType::kFLOAT, &weights[index], channels[channelIndex]};
            index += channels[channelIndex];

            Weights layerBias{DataType::kFLOAT, &weights[index], channels[channelIndex]};
            index += channels[channelIndex];

            Weights layerPower{DataType::kFLOAT, &weights[index], channels[channelIndex]};
            index += 2 * channels[channelIndex];

            auto scaleLayer = network->addScale(*convLayer->getOutput(0), ScaleMode::kCHANNEL, layerBias, layerScale,
                                                layerPower);

            auto activationLayer = network->addActivation(*scaleLayer->getOutput(0), ActivationType::kRELU);
            prevLayer = activationLayer->getOutput(0);
        }

        if (channelIndex == 6) {
            prevLayers.push_back(prevLayer);
        }
        if (channelIndex < channels.size() - 1) {
            auto poolLayer = network->addPooling(*prevLayer, PoolingType::kMAX, DimsHW{2, 2});
            poolLayer->setStride(DimsHW{2, 2});
            prevLayer = poolLayer->getOutput(0);
        }
    }
    for (int channelIndex = 0; channelIndex < prevLayers.size(); channelIndex++) {

        prevLayer = prevLayers[channelIndex];
        int convWeightsCount = prevLayer->getDimensions().d[0] * COORDINATE_SIZES[channelIndex];
        Weights convWeights{DataType::kFLOAT, &weights[index], convWeightsCount};
        index += convWeightsCount;

        int convBiasesCount = COORDINATE_SIZES[channelIndex];
        Weights convBias{DataType::kFLOAT, &weights[index], convBiasesCount};
        index += convBiasesCount;

        auto convLayer = network->addConvolution(*prevLayer, COORDINATE_SIZES[channelIndex], DimsHW{1, 1}, convWeights,
                                                 convBias);

        convLayer->getOutput(0)->setName(NETWORK_OUTPUT_NAMES[channelIndex].c_str());
        network->markOutput(*convLayer->getOutput(0));
    }

    unique_ptr<IBuilderConfig, TensorRTDeleter> builderConfig(builder->createBuilderConfig(), TensorRTDeleter());

    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    builderConfig->setMaxWorkspaceSize(1 << 30);

    engine = builder->buildEngineWithConfig(*network, *builderConfig);
}


nvinfer1::IExecutionContext *CarDetectionEngine::createExecutionContext() {
    return engine->createExecutionContext();
}