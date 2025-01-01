#pragma once

template<typename T>

class Counter {
public:
    explicit Counter(T item) : item{move(item)} {};

    void incrementOccurrence() {
        numberOfOccurrence += 1;
    }

    int getNumberOfOccurrence() const {
        return numberOfOccurrence;
    }

    const T &getItem() const {
        return item;
    }

    void renewItem(T copyItem) {
        item = move(copyItem);
    }

    void insertIntoDet(float prob) {
        detProbabilities.push_back(prob);
    }

    void insertIntoRec(float prob) {
        recProbabilities.push_back(prob);
    }

    [[nodiscard]] float getDetAverageProb() const {
        auto const count = static_cast<float>(detProbabilities.size());
        return std::reduce(detProbabilities.begin(), detProbabilities.end()) / count;
    }

    [[nodiscard]] float getRecAverageProb() const {
        auto const count = static_cast<float>(recProbabilities.size());
        return std::reduce(recProbabilities.begin(), recProbabilities.end()) / count;
    }

private:
    T item;
    int numberOfOccurrence = 1;
    std::vector<float> detProbabilities, recProbabilities;
};

