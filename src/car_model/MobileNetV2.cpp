//
// Created by kartykbayev on 11/2/23.
//

#include "MobileNetV2.h"

#include <utility>

using namespace std;
using namespace nvinfer1;

MobileNetV2::MobileNetV2(Constants::VehicleClassificationType type) {

    ENGINE_NAME = Constants::CLS_TYPE_TO_ENGINE_NAME.find(type)->second;
    WEIGHTS_FILENAME = Constants::CLS_TYPE_TO_WEIGHTS_NAME.find(type)->second;

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

nvinfer1::IExecutionContext *MobileNetV2::createExecutionContext() {
    return engine->createExecutionContext();
}

void MobileNetV2::createEngine() {
    TrtLogger trt_logger;
    auto builder = unique_ptr<IBuilder, TensorRTDeleter>(createInferBuilder(trt_logger), TensorRTDeleter());
    vector<float> weights;
    ifstream weightFile(Constants::modelWeightsFolder + WEIGHTS_FILENAME, ios::binary);

    float parameter;

    weights.reserve(weightFile.tellg() / 4); // char to float

    while (weightFile.read(reinterpret_cast<char *>(&parameter), sizeof(float))) {
        weights.push_back(parameter);
    }

    auto network = unique_ptr<INetworkDefinition, TensorRTDeleter>(
            builder->createNetworkV2(0), TensorRTDeleter());

    ITensor *inputLayer = network->addInput(NETWORK_INPUT_NAME.c_str(), DataType::kFLOAT,
                                            Dims3{IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS});
    IShuffleLayer *shuffle = network->addShuffle(*inputLayer);
    shuffle->setFirstTranspose(Permutation{2, 0, 1});
    shuffle->setReshapeDimensions(Dims3{IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT});

    int index = 0;
    int firstLayerChannels = 32;
    int lastLayerChannels = 1280;
    int width = 7;
    vector<vector<int>> networkConfig{
            {1, 16,  1, 1},
            {6, 24,  2, 2},
            {6, 32,  3, 2},
            {6, 64,  4, 2},
            {6, 96,  3, 1},
            {6, 160, 3, 2},
            {6, 320, 1, 1}
    };

    auto prevLayer = convBnRelu(shuffle->getOutput(0), network.get(), index, weights,
                                firstLayerChannels, 3, 2, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 32, 16, 1, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 16, 24, 6, 2);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 24, 24, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 24, 32, 6, 2);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 32, 32, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 32, 32, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 32, 64, 6, 2);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 64, 64, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 64, 64, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 64, 64, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 64, 96, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 96, 96, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 96, 96, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 96, 160, 6, 2);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 160, 160, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 160, 160, 6, 1);
    prevLayer = invertedResidual(prevLayer, network.get(), index, weights, 160, 320, 6, 1);
    prevLayer = convBnRelu(prevLayer, network.get(), index, weights, lastLayerChannels, 1, 1, 1);
    prevLayer = network->addPooling(*prevLayer, PoolingType::kAVERAGE, DimsHW{width, width})->getOutput(0);
    auto fcWeights = Weights{DataType::kFLOAT, &weights[index], lastLayerChannels * OUTPUT_SIZE};
    index += lastLayerChannels * OUTPUT_SIZE;
    auto fcBiases = Weights{DataType::kFLOAT, &weights[index], OUTPUT_SIZE};
    index += OUTPUT_SIZE;
    auto fc = network->addFullyConnected(*prevLayer, OUTPUT_SIZE, fcWeights, fcBiases)->getOutput(0);
    network->markOutput(*fc);

    unique_ptr<IBuilderConfig, TensorRTDeleter> builderConfig(builder->createBuilderConfig(), TensorRTDeleter());

    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    builderConfig->setMaxWorkspaceSize(1 << 30);

    engine = builder->buildEngineWithConfig(*network, *builderConfig);
}

nvinfer1::ITensor *
MobileNetV2::convBnRelu(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network, int &index,
                        vector<float> &weights, int outChannels, int kernelSize, int strideStep,
                        int numGroups) {
    Weights emptyWeights{DataType::kFLOAT, nullptr, 0};
    int paddingSize = (kernelSize - 1) / 2;
    DimsHW padding{paddingSize, paddingSize};
    DimsHW kernel{kernelSize, kernelSize};
    DimsHW stride{strideStep, strideStep};

    int convWeightsCount = (inputTensor->getDimensions().d[0] / numGroups) * outChannels * kernelSize * kernelSize;
    auto convWeights = Weights{DataType::kFLOAT, &weights[index], convWeightsCount};
    index += convWeightsCount;

    auto convLayer = network->addConvolution(*inputTensor, outChannels, kernel, convWeights, emptyWeights);
    convLayer->setStride(stride);
    convLayer->setPadding(padding);
    convLayer->setNbGroups(numGroups);

    assert(convLayer);

    auto batchLayer = batchNorm(convLayer->getOutput(0), network, index, weights,
                                outChannels);

    assert(batchLayer);

    auto relu1 = network->addActivation(*batchLayer, ActivationType::kRELU);
    assert(relu1);
    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * 1));
    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * 1));
    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * 1));
    shval[0] = -6.0;
    scval[0] = 1.0;
    pval[0] = 1.0;
    Weights shift{DataType::kFLOAT, shval, 1};
    Weights scale{DataType::kFLOAT, scval, 1};
    Weights power{DataType::kFLOAT, pval, 1};

    auto scale1 = network->addScale(*batchLayer, ScaleMode::kUNIFORM, shift, scale, power);
    assert(scale1);
    auto relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IElementWiseLayer *ew1 = network->addElementWise(*relu1->getOutput(0), *relu2->getOutput(0),
                                                     ElementWiseOperation::kSUB);
    assert(ew1);
    return ew1->getOutput(0);
}

nvinfer1::ITensor *
MobileNetV2::batchNorm(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network, int &index,
                       vector<float> &weights, int channels) {
    for (int channel = 0; channel < channels; channel++) {

        weights[index + channel] /= sqrt(weights[index + channels * 3 + channel] + 1e-5);

        weights[index + channels + channel] -=
                weights[index + channels * 2 + channel] * weights[index + channel];

        weights[index + channels * 2 + channel] = 1.0;
    }

    auto layerScale = Weights{DataType::kFLOAT, &weights[index], channels};
    index += channels;

    auto layerBias = Weights{DataType::kFLOAT, &weights[index], channels};
    index += channels;

    auto layerPower = Weights{DataType::kFLOAT, &weights[index], channels};
    index += 2 * channels;

    auto scaleLayer = network->addScale(*inputTensor, ScaleMode::kCHANNEL,
                                        layerBias, layerScale, layerPower);
    assert(scaleLayer);
    return scaleLayer->getOutput(0);
}

nvinfer1::ITensor *
MobileNetV2::invertedResidual(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network, int &index,
                              vector<float> &weights, int inChannels, int outChannels, int expandRatio,
                              int strideStep) {

    int hiddenDim = inChannels * expandRatio;
    bool useResidual = strideStep == 1 && inChannels == outChannels;

    if (expandRatio != 1) {
        auto ew1 = convBnRelu(inputTensor, network, index, weights, hiddenDim, 1, 1, 1);
        auto ew2 = convBnRelu(ew1, network, index, weights, hiddenDim, 3, strideStep,
                              hiddenDim);
        auto convWeightsCount = ew2->getDimensions().d[0] * outChannels * 1 * 1;
        auto convWeights = Weights{DataType::kFLOAT, &weights[index], convWeightsCount};
        index += convWeightsCount;
        auto convBias = Weights{DataType::kFLOAT, nullptr, 0};
        auto resConvLayer = network->addConvolution(*ew2, outChannels, DimsHW{1, 1}, convWeights, convBias);
        auto bn1 = batchNorm(resConvLayer->getOutput(0), network, index, weights,
                             outChannels);
        if (!useResidual) return bn1;

        return network->addElementWise(*inputTensor, *bn1, ElementWiseOperation::kSUM)->getOutput(0);
    } else {
        auto ew1 = convBnRelu(inputTensor, network, index, weights, hiddenDim, 3,
                              strideStep,
                              hiddenDim);
        auto convWeightsCount = ew1->getDimensions().d[0] * outChannels * 1 * 1;
        auto convWeights = Weights{DataType::kFLOAT, &weights[index], convWeightsCount};
        index += convWeightsCount;
        auto convBias = Weights{DataType::kFLOAT, nullptr, 0};
        auto resConvLayer = network->addConvolution(*ew1, outChannels, DimsHW{1, 1}, convWeights, convBias);
        auto bn1 = batchNorm(resConvLayer->getOutput(0), network, index, weights,
                             outChannels);

        if (!useResidual) return bn1;

        return network->addElementWise(*inputTensor, *bn1, ElementWiseOperation::kSUM)->getOutput(0);
    }

}
