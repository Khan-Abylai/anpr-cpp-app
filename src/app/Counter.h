#pragma once

template<typename T>

class Counter {
public:
    explicit Counter(T item) : item{move(item)} {};

    void incrementOccurrence() {
        numberOfOccurrence += 1;
    }

    [[nodiscard]] int getNumberOfOccurrence() const {
        return numberOfOccurrence;
    }

    const T &getItem() const {
        return item;
    }

    void renewItem(T copyItem) {
        item = move(copyItem);
    }

private:
    T item;
    int numberOfOccurrence = 1;
    std::vector<float> detProbabilities, recProbabilities;
};

