//
// Created by artyk on 11/9/2023.
//

#include "MobileNet.h"

using namespace std;
using namespace nvinfer1;


MobileNet::MobileNet() {
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

nvinfer1::IExecutionContext *MobileNet::createExecutionContext() {
    return engine->createExecutionContext();
}

void MobileNet::createEngine() {
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
    shuffle->setFirstTranspose(Permutation{2, 1, 0});
    shuffle->setReshapeDimensions(Dims3{IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT});

    float initialScale = 1.f / 255;
    float initialBias = 0;
    float initialPower = 1;
    auto scaleLayer = network->addScale(*shuffle->getOutput(0),
                                        ScaleMode::kUNIFORM,
                                        Weights{DataType::kFLOAT, &initialBias, 1},
                                        Weights{DataType::kFLOAT, &initialScale, 1},
                                        Weights{DataType::kFLOAT, &initialPower, 1});

    int index = 0;
    int firstLayerChannels = 32;
    int lastLayerChannels = 1280;

    vector<vector<int>> networkConfig{
            {1, 16,  1, 1},
            {6, 24,  2, 2},
            {6, 32,  3, 2},
            {6, 64,  4, 2},
            {6, 96,  3, 1},
            {6, 160, 3, 2},
            {6, 320, 1, 1}
    };

    auto prevLayer = convBnRelu(scaleLayer->getOutput(0), network.get(), index, weights, IMG_CHANNELS,
                                firstLayerChannels, 3, 2, 1);
    int inChannels = firstLayerChannels;

    for (auto &layerConfig: networkConfig) {
        int expandRatio = layerConfig[0];
        int channels = layerConfig[1];
        int numLayers = layerConfig[2];
        int stride = layerConfig[3];

        for (int i = 0; i < numLayers; i++) {
            prevLayer = invertedResidual(prevLayer, network.get(), index, weights, inChannels, channels,
                                         expandRatio,
                                         (i == 0) ? stride : 1);
            inChannels = channels;
        }
    }

    prevLayer = convBnRelu(prevLayer, network.get(), index, weights, inChannels, lastLayerChannels, 1, 1, 1);
    int width = prevLayer->getDimensions().d[1];
    prevLayer = network->addPooling(*prevLayer, PoolingType::kAVERAGE, DimsHW{width, width})->getOutput(0);

    auto fcWeights = Weights{DataType::kFLOAT, &weights[index], lastLayerChannels * OUTPUT_SIZE};
    index += lastLayerChannels * OUTPUT_SIZE;
    auto fcBiases = Weights{DataType::kFLOAT, &weights[index], OUTPUT_SIZE};
    index += OUTPUT_SIZE;

    network->markOutput(*network->addFullyConnected(*prevLayer, OUTPUT_SIZE, fcWeights, fcBiases)->getOutput(0));

    unique_ptr<IBuilderConfig, TensorRTDeleter> builderConfig(builder->createBuilderConfig(), TensorRTDeleter());

    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    builderConfig->setMaxWorkspaceSize(1 << 30);

    engine = builder->buildEngineWithConfig(*network, *builderConfig);

}

nvinfer1::ITensor *
MobileNet::batchNorm(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network, int &index,
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
    return scaleLayer->getOutput(0);

}

nvinfer1::ITensor *
MobileNet::convBnRelu(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network, int &index,
                      vector<float> &weights, int inChannels, int outChannels, int kernelSize, int strideStep,
                      int numGroups) {

    int paddingSize = (kernelSize == 1) ? 0 : 1;
    DimsHW kernel{kernelSize, kernelSize};
    DimsHW stride{strideStep, strideStep};

    auto padding = DimsHW{paddingSize, paddingSize};

    int convWeightsCount = inChannels * outChannels * kernelSize * kernelSize / numGroups;
    auto convWeights = Weights{DataType::kFLOAT, &weights[index], convWeightsCount};

    index += convWeightsCount;

    auto convBias = Weights{DataType::kFLOAT, nullptr, 0};
    auto convLayer = network->addConvolution(*inputTensor, outChannels, kernel, convWeights, convBias);

    convLayer->setStride(stride);
    convLayer->setPadding(padding);
    convLayer->setNbGroups(numGroups);

    auto batchLayer = batchNorm(convLayer->getOutput(0), network, index, weights, outChannels);

    return network->addActivation(*batchLayer, ActivationType::kRELU)->getOutput(0);

}

nvinfer1::ITensor *
MobileNet::invertedResidual(nvinfer1::ITensor *inputTensor, nvinfer1::INetworkDefinition *network, int &index,
                            vector<float> &weights, int inChannels, int outChannels, int expandRatio, int strideStep) {
    auto residualLayer = inputTensor;
    int hiddenDim = inChannels * expandRatio;
    bool useResidual = strideStep == 1 && inChannels == outChannels;

    if (expandRatio != 1) {
        residualLayer = convBnRelu(residualLayer, network, index, weights, inChannels, hiddenDim, 1, 1, 1);
    }

    auto residualLayer2 = convBnRelu(residualLayer, network, index, weights, hiddenDim, hiddenDim, 3, strideStep,
                                     hiddenDim);
    auto convWeightsCount = hiddenDim * outChannels;
    auto convWeights = Weights{DataType::kFLOAT, &weights[index], convWeightsCount};
    auto convBias = Weights{DataType::kFLOAT, nullptr, 0};

    auto resConvLayer = network->addConvolution(*residualLayer2, outChannels, DimsHW{1, 1}, convWeights, convBias);

    resConvLayer->setStride(DimsHW{1, 1,});
    resConvLayer->setPadding(DimsHW{0, 0});

    index += convWeightsCount;

    auto batchLayer = batchNorm(resConvLayer->getOutput(0), network, index, weights, outChannels);

    if (!useResidual) {
        return batchLayer;
    }

    return network->addElementWise(*batchLayer, *inputTensor, ElementWiseOperation::kSUM)->getOutput(0);

}
