#include "LPRecognizer.h"

using namespace std;
using namespace nvinfer1;

LPRecognizer::LPRecognizer() {

    if (!filesystem::exists(ENGINE_NAME)) {
        createEngine();
        TensorRTEngine::serializeEngine(engine, ENGINE_NAME);
    }

    engine = TensorRTEngine::readEngine(ENGINE_NAME);
    dimensions.resize(MAX_BATCH_SIZE * SEQUENCE_SIZE, ALPHABET.size());

    cudaMalloc(&cudaBuffer[0], MAX_BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&cudaBuffer[1], MAX_BATCH_SIZE * SEQUENCE_SIZE * sizeof(int));
    cudaMalloc(&cudaBuffer[2], MAX_BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    // cudaMalloc(&cudaBuffer[3], MAX_BATCH_SIZE * OUTPUT_2_SIZE * sizeof(float));

    cudaMemcpy(cudaBuffer[1], dimensions.data(), MAX_BATCH_SIZE * SEQUENCE_SIZE * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream);

    executionContext = engine->createExecutionContext();
}

LPRecognizer::~LPRecognizer() {

    cudaFree(cudaBuffer[0]);
    cudaFree(cudaBuffer[1]);
    cudaFree(cudaBuffer[2]);
    // cudaFree(cudaBuffer[3]);
    cudaStreamDestroy(stream);
}

void LPRecognizer::createEngine() {

    auto builder = unique_ptr<IBuilder, TensorRTDeleter>(createInferBuilder(TensorRTEngine::getLogger()),
                                                         TensorRTDeleter());

    vector<float> weights;
    ifstream weightFile(Constants::modelWeightsFolder + WEIGHTS_FILENAME, ios::binary);

    float parameter;

    weights.reserve(weightFile.tellg() / 4); // char to float

    while (weightFile.read(reinterpret_cast<char *>(&parameter), sizeof(float))) {
        weights.push_back(parameter);
    }

    auto network = unique_ptr<INetworkDefinition, TensorRTDeleter>(
            builder->createNetworkV2(0),
            TensorRTDeleter());
    ITensor *inputLayer = network->addInput(NETWORK_INPUT_NAME.c_str(), DataType::kFLOAT,
                                            Dims3{IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH});

    ITensor *prevLayer;
    vector<int> channels{128, 256, 512};

    DimsHW kernelSize{3, 3};
    int hiddenSize = 256;
    int index = 0;

    prevLayer = inputLayer;

    for (int i = 0; i < channels.size(); i++) {
        for (int j = 0; j < 3; j++) {

            int convWeightsCount =
                    prevLayer->getDimensions().d[0] * channels[i] * kernelSize.d[0] * kernelSize.d[1];
            Weights convWeights{DataType::kFLOAT, &weights[index], convWeightsCount};
            index += convWeightsCount;

            int convBiasesCount = channels[i];
            Weights convBias{DataType::kFLOAT, &weights[index], convBiasesCount};
            index += convBiasesCount;

            auto convLayer = network->addConvolution(*prevLayer, channels[i], kernelSize, convWeights, convBias);
            convLayer->setStride(DimsHW{1, 1});

            if (i == 2 && j == 2) {
                convLayer->setPadding(DimsHW{0, 0});
            } else {
                convLayer->setPadding(DimsHW{1, 1});
            }

            for (int k = 0; k < channels[i]; k++) {
                weights[index + k] /= sqrt(weights[index + channels[i] * 3 + k] + 1e-5);
                weights[index + channels[i] + k] -= weights[index + channels[i] * 2 + k] * weights[index + k];
                weights[index + channels[i] * 2 + k] = 1.0;
            }

            Weights scaleLayerWeights{DataType::kFLOAT, &weights[index], channels[i]};
            index += channels[i];

            Weights biasLayerWeights{DataType::kFLOAT, &weights[index], channels[i]};
            index += channels[i];

            Weights powerLayerWeights{DataType::kFLOAT, &weights[index], channels[i]};
            index += 2 * channels[i];

            auto scaleLayer = network->addScale(*convLayer->getOutput(0), ScaleMode::kCHANNEL,
                                                biasLayerWeights, scaleLayerWeights, powerLayerWeights);

            auto activLayer = network->addActivation(*scaleLayer->getOutput(0), ActivationType::kRELU);
            prevLayer = activLayer->getOutput(0);
        }
        if (i < channels.size() - 1) {
            auto poolLayer = network->addPooling(*prevLayer, PoolingType::kMAX, DimsHW{2, 2});
            poolLayer->setStride(DimsHW{2, 2});
            prevLayer = poolLayer->getOutput(0);
        }
    }
    auto size = prevLayer->getDimensions();
    IShuffleLayer *shuffle = network->addShuffle(*prevLayer);
    shuffle->setReshapeDimensions(Dims2{prevLayer->getDimensions().d[0] * prevLayer->getDimensions().d[1],
                                        prevLayer->getDimensions().d[2]});
    shuffle->setSecondTranspose(Permutation{1, 0});

    Dims2 embeddingShape = Dims2(hiddenSize, shuffle->getOutput(0)->getDimensions().d[1]);

    Weights embeddingWeights{DataType::kFLOAT, &weights[index],
                             hiddenSize * shuffle->getOutput(0)->getDimensions().d[1]};

    index += hiddenSize * shuffle->getOutput(0)->getDimensions().d[1];

    auto embedding = network->addConstant(embeddingShape, embeddingWeights);

    IMatrixMultiplyLayer *matrixMultiplication1 = network->addMatrixMultiply(*shuffle->getOutput(0),
                                                                             MatrixOperation::kNONE,
                                                                             *embedding->getOutput(0),
                                                                             MatrixOperation::kTRANSPOSE);

    Dims2 embeddingBiasShape = Dims2(1, hiddenSize);

    Weights embeddingBiasWeights{DataType::kFLOAT, &weights[index], hiddenSize};
    index += hiddenSize;

    auto embeddingBias = network->addConstant(embeddingBiasShape, embeddingBiasWeights);

    auto addBias = network->addElementWise(*matrixMultiplication1->getOutput(0), *embeddingBias->getOutput(0),
                                           ElementWiseOperation::kSUM);

    prevLayer = network->addActivation(*addBias->getOutput(0), ActivationType::kRELU)->getOutput(0);

    auto newShuffle = network->addShuffle(*prevLayer);
    newShuffle->setReshapeDimensions(Dims3{prevLayer->getDimensions().d[0] * prevLayer->getDimensions().d[1], 1, 1});

    // auto CountryClsWeights = Weights{DataType::kFLOAT, &weights[index],
    //                                  prevLayer->getDimensions().d[0] * prevLayer->getDimensions().d[1] * REGION_SIZE};
    // index += prevLayer->getDimensions().d[0] * prevLayer->getDimensions().d[1] * REGION_SIZE;

    // auto CountryClsBias = Weights{DataType::kFLOAT, &weights[index], REGION_SIZE};
    // index += REGION_SIZE;
    // auto CountryLayer = network->addFullyConnected(
    //         *newShuffle->getOutput(0), REGION_SIZE,
    //         CountryClsWeights, CountryClsBias
    // );

    int lstmLayerCount = 2;

    int lstmInputSize = hiddenSize;
    IRNNv2Layer *lstm = network->addRNNv2(*prevLayer, lstmLayerCount, hiddenSize, SEQUENCE_SIZE, RNNOperation::kLSTM);
    vector<RNNGateType> gates{RNNGateType::kINPUT, RNNGateType::kFORGET, RNNGateType::kCELL,
                              RNNGateType::kOUTPUT};
    lstm->setDirection(nvinfer1::RNNDirection::kBIDIRECTION);
    int hidden2 = (int) pow(hiddenSize, 2);
    int hidden22 = lstmInputSize * hiddenSize;


    for (int layerIndex = 0; layerIndex < lstmLayerCount; ++layerIndex) {
        int rela_index = 2 * layerIndex;
        int iw_count;
        if (layerIndex == 1)
            iw_count = hidden22 * 2;
        else
            iw_count = hidden22;
        for (int i = 0; i < gates.size(); ++i) {

            lstm->setWeightsForGate(rela_index, gates[i], true,
                                    Weights{DataType::kFLOAT, &weights[index + i * iw_count], iw_count});
        }
        index += gates.size() * iw_count;
        for (int i = 0; i < gates.size(); i++) {
            lstm->setWeightsForGate(rela_index, gates[i], false,
                                    Weights{DataType::kFLOAT, &weights[index + i * hidden2], hidden2});
        }
        index += hiddenSize * gates.size() * hiddenSize;

        for (int i = 0; i < gates.size(); i++) {
            lstm->setBiasForGate(rela_index, gates[i], true,
                                 Weights{DataType::kFLOAT, &weights[index + i * hiddenSize], hiddenSize});
        }
        index += hiddenSize * gates.size();

        for (int i = 0; i < gates.size(); i++) {
            lstm->setBiasForGate(rela_index, gates[i], false,
                                 Weights{DataType::kFLOAT, &weights[index + i * hiddenSize], hiddenSize});
        }
        index += hiddenSize * gates.size();


        rela_index = 2 * layerIndex + 1;

        for (int i = 0; i < gates.size(); ++i) {

            lstm->setWeightsForGate(rela_index, gates[i], true,
                                    Weights{DataType::kFLOAT, &weights[index + i * iw_count], iw_count});
        }
        index += gates.size() * iw_count;
        for (int i = 0; i < gates.size(); i++) {
            lstm->setWeightsForGate(rela_index, gates[i], false,
                                    Weights{DataType::kFLOAT, &weights[index + i * hidden2], hidden2});
        }
        index += hiddenSize * gates.size() * hiddenSize;

        for (int i = 0; i < gates.size(); i++) {
            lstm->setBiasForGate(rela_index, gates[i], true,
                                 Weights{DataType::kFLOAT, &weights[index + i * hiddenSize], hiddenSize});
        }
        index += hiddenSize * gates.size();

        for (int i = 0; i < gates.size(); i++) {
            lstm->setBiasForGate(rela_index, gates[i], false,
                                 Weights{DataType::kFLOAT, &weights[index + i * hiddenSize], hiddenSize});
        }
        index += hiddenSize * gates.size();

    }

    Dims embeddingShape2 = Dims2(ALPHABET_SIZE, hiddenSize * 2);
    Weights embeddingWeights2{DataType::kFLOAT, &weights[index], embeddingShape2.d[0] * embeddingShape2.d[1]};
    index += embeddingShape2.d[0] * embeddingShape2.d[1];
    IConstantLayer *embedding2 = network->addConstant(embeddingShape2, embeddingWeights2);
    IMatrixMultiplyLayer *matrixMultiplication2 = network->addMatrixMultiply(*lstm->getOutput(0),
                                                                             MatrixOperation::kNONE,
                                                                             *embedding2->getOutput(0),
                                                                             MatrixOperation::kTRANSPOSE);
    Dims embeddingBiasShape2 = Dims2(1, ALPHABET_SIZE);
    Weights embeddingBiasWeights2 = Weights{DataType::kFLOAT, &weights[index], ALPHABET_SIZE};

    IConstantLayer *bias2 = network->addConstant(embeddingBiasShape2, embeddingBiasWeights2);
    IElementWiseLayer *addBias2 = network->addElementWise(*matrixMultiplication2->getOutput(0), *bias2->getOutput(0),
                                                          ElementWiseOperation::kSUM);

    ITensor *dimensions = network->addInput(NETWORK_DIM_NAME.c_str(), DataType::kINT32, Dims2(SEQUENCE_SIZE, 1));
    IRaggedSoftMaxLayer *softmax = network->addRaggedSoftMax(*addBias2->getOutput(0), *dimensions);

    // ISoftMaxLayer *countrySoftmax = network->addSoftMax(*CountryLayer->getOutput(0));


    // network->markOutput(*countrySoftmax->getOutput(0));
    network->markOutput(*softmax->getOutput(0));

    unique_ptr<IBuilderConfig, TensorRTDeleter> builderConfig(builder->createBuilderConfig(), TensorRTDeleter());

    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    builderConfig->setMaxWorkspaceSize(1 << 30);

    engine = builder->buildEngineWithConfig(*network, *builderConfig);
}

vector<float> LPRecognizer::prepareImage(const vector<cv::Mat> &frames)  {

    int batchSize = frames.size();

    vector<float> flattenedImage;
    flattenedImage.resize(batchSize * INPUT_SIZE);
    cv::Mat resizedFrame;

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        resize(frames[batchIndex], resizedFrame, cv::Size(IMG_WIDTH, IMG_HEIGHT));

        for (int row = 0; row < resizedFrame.rows; row++) {
            for (int col = 0; col < resizedFrame.cols; col++) {
                uchar *pixels = resizedFrame.data + resizedFrame.step[0] * row + resizedFrame.step[1] * col;
                flattenedImage[batchIndex * 3 * IMG_HEIGHT * IMG_WIDTH + row * IMG_WIDTH + col] =
                        static_cast<float>(pixels[0]) / Constants::PIXEL_MAX_VALUE;

                flattenedImage[batchIndex * 3 * IMG_HEIGHT * IMG_WIDTH + row * IMG_WIDTH + col +
                               IMG_HEIGHT * IMG_WIDTH] =
                        static_cast<float>(pixels[1]) / Constants::PIXEL_MAX_VALUE;

                flattenedImage[batchIndex * 3 * IMG_HEIGHT * IMG_WIDTH + row * IMG_WIDTH + col +
                               2 * IMG_HEIGHT * IMG_WIDTH] =
                        static_cast<float>(pixels[2]) / Constants::PIXEL_MAX_VALUE;
            }
        }
    }

    return move(flattenedImage);
}

std::tuple<std::string, double> LPRecognizer::makePrediction(const vector<cv::Mat> &frames) {

    int batchSize = frames.size();

    auto predictions = executeInferEngine(frames);

    // auto clsPredictions = engineExecResult.first;
    // auto predictions = engineExecResult.second;

    tuple<string, float> label;

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {

        double prob = 1.0;
        string currentLabel;
        currentLabel.reserve(MAX_PLATE_SIZE);

        int currentChar = BLANK_INDEX;
        float currentProb = 1.0;

        for (int i = 0; i < SEQUENCE_SIZE; i++) {

            float maxProb = 0.0;
            int maxIndex = 0;

            for (int j = 0; j < ALPHABET_SIZE; j++) {
                if (maxProb < predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j]) {
                    maxIndex = j;
                    maxProb = predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j];
                }
            }

            if (maxIndex == currentChar) {
                currentProb = max(maxProb, currentProb);
            } else {
                if (currentChar != BLANK_INDEX) {
                    currentLabel += ALPHABET[currentChar];
                    prob *= currentProb;
                }
                currentProb = maxProb;
                currentChar = maxIndex;
            }
        }

        if (currentChar != BLANK_INDEX) {
            currentLabel += ALPHABET[currentChar];
            prob *= currentProb;
        }

        if (currentLabel.empty()) {
            currentLabel += ALPHABET[BLANK_INDEX];
            prob = 0.0;
        }
        // int maxElementIndex = std::max_element(clsPredictions.begin(), clsPredictions.end()) - clsPredictions.begin();
        label = make_tuple(currentLabel, prob);
    }

    return label;
}

std::vector<float> LPRecognizer::executeInferEngine(const vector<cv::Mat> &frames) {
    auto flattenedImages = prepareImage(frames);
    auto stop = 1;
    int batchSize = frames.size();

    vector<float> predictions;//, predictions2;
    predictions.resize(batchSize * OUTPUT_SIZE);
    // predictions2.resize(batchSize * OUTPUT_2_SIZE);

    cudaMemcpyAsync(cudaBuffer[0], flattenedImages.data(), batchSize * INPUT_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    executionContext->enqueue(batchSize, cudaBuffer, stream, nullptr);

    cudaMemcpyAsync(predictions.data(), cudaBuffer[2], batchSize * OUTPUT_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(predictions2.data(), cudaBuffer[3], batchSize * OUTPUT_2_SIZE * sizeof(float),
    //                 cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    return predictions;
}
