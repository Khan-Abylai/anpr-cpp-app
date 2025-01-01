#include <filesystem>
#include <fstream>
#include <vector>

#include "TensorRTEngine.h"
#include "TrtLogger.h"
#include "TensorRTDeleter.h"

using namespace std;
using namespace nvinfer1;

void TensorRTEngine::serializeEngine(ICudaEngine *engine, const string &engineFilename) {

    ofstream engineFile(engineFilename, ios::binary);
    unique_ptr<IHostMemory, TensorRTDeleter> trtModelStream{engine->serialize(), TensorRTDeleter()};
    engineFile.write((char *) trtModelStream->data(), (int)trtModelStream->size());
}

ICudaEngine *TensorRTEngine::readEngine(const string &engineFilename) {

    ifstream engineFile(engineFilename);

    engineFile.seekg(0, ios::end);
    const int modelSize =(int) engineFile.tellg();
    engineFile.seekg(0, ios::beg);

    vector<char> engineData(modelSize);
    engineFile.read(engineData.data(), modelSize);

    TrtLogger trt_logger;
    auto infer = unique_ptr<IRuntime, TensorRTDeleter>(createInferRuntime(trt_logger), TensorRTDeleter());

    return infer->deserializeCudaEngine(engineData.data(), modelSize, nullptr);
}