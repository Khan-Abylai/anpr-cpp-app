//
// Created by artyk on 11/9/2023.
//

#include "VehicleRecognizer.h"


using namespace std;

std::vector<float> VehicleRecognizer::softmax(std::vector<float> &score_vec) {
    vector<float> softmax_vec(score_vec.size());
    double score_max = *(max_element(score_vec.begin(), score_vec.end()));
    double e_sum = 0;
    for (int j = 0; j < score_vec.size(); j++) {
        softmax_vec[j] = exp((double) score_vec[j] - score_max);
        e_sum += softmax_vec[j];
    }
    for (int k = 0; k < score_vec.size(); k++)
        softmax_vec[k] /= e_sum;
    return softmax_vec;
}

std::pair<std::string, double> VehicleRecognizer::classify(const shared_ptr<LicensePlate> &licensePlate) {
    return {};
}
