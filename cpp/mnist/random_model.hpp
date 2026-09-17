#pragma once

#include <random>

#include "digit_classification_interface.hpp"

// Expects a flat vector of 100 values -- a 10x10 center crop.
class RandomModel : public DigitClassificationInterface {
public:
    static constexpr size_t kInputLength = 10 * 10;

    explicit RandomModel(unsigned seed = 0);

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) override;
    int predict(const std::vector<double>& image) override;

private:
    std::mt19937 rng_;
    std::uniform_int_distribution<int> dist_;
};
