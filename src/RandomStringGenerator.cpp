#include "RandomStringGenerator.h"

using namespace std;

const string RandomStringGenerator::ALPHA_NUM =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

uniform_int_distribution<int>  RandomStringGenerator::randomGenerator(0, ALPHA_NUM.length() - 1);

string RandomStringGenerator::generate(int length) {

    string filename;
    filename.resize(length);
    random_device rd;
    mt19937 generator(rd());
    for (char &c: filename) {
        c = ALPHA_NUM[randomGenerator(generator)];
    }
    return move(filename);
}

string RandomStringGenerator::generate(int length, const string &prepend, const string &append) {
    string filename;
    filename.resize(prepend.size() + length + append.size());
    random_device rd;
    mt19937 generator(rd());

    int i = 0;

    for (auto prependChar: prepend) {
        filename[i++] = prependChar;
    }

    for (int j = 0; j < length; j++) {
        filename[i++] = ALPHA_NUM[randomGenerator(generator)];
    }

    for (auto appendChar: append) {
        filename[i++] = appendChar;
    }

    return move(filename);
}