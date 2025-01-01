//
// Created by kartykbayev on 6/6/22.
//

#include "TensorRTEngine.h"

using namespace std;
using namespace nvinfer1;

TrtLogger TensorRTEngine::trtLogger;

void TensorRTEngine::serializeEngine(nvinfer1::ICudaEngine *engine, const string &engineFilename) {
    ofstream engineFile(engineFilename, ios::binary);
    unique_ptr<IHostMemory, TensorRTDeleter> trtModelStream{engine->serialize(), TensorRTDeleter()};
    engineFile.write((char *) trtModelStream->data(), trtModelStream->size());
}

nvinfer1::ICudaEngine *TensorRTEngine::readEngine(const string &engineFilename) {
    ifstream inEngineFile(engineFilename);

    inEngineFile.seekg(0, ios::end);
    const int modelSize = inEngineFile.tellg();
    inEngineFile.seekg(0, ios::beg);

    vector<char> engineData(modelSize);
    inEngineFile.read(engineData.data(), modelSize);

    auto infer = unique_ptr<IRuntime, TensorRTDeleter>(nvinfer1::createInferRuntime(getLogger()), TensorRTDeleter());

    return infer->deserializeCudaEngine(engineData.data(), modelSize, nullptr);

}
