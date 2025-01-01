#pragma once

#include <string>
#include <random>


class RandomStringGenerator {
public:

    static std::string generate(int length);

    static std::string generate(int length, const std::string &prepend, const std::string &append);

private:

    static const std::string ALPHA_NUM;
    static std::uniform_int_distribution<int> randomGenerator;
};

