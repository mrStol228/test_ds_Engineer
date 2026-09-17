#pragma once

#include <random>

#include "digit_classification_interface.hpp"
#include "layers.hpp"

// Expects a flat vector of 784 values -- a 28x28 image, row-major, one channel.
class CnnModel : public DigitClassificationInterface {
public:
    static constexpr size_t kInputLength = 28 * 28;

    CnnModel();

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) override;
    int predict(const std::vector<double>& image) override;

private:
    // Declaration order IS initialization order in C++ (the initializer
    // list order below is ignored for this) -- rng_ must come first so it
    // exists before conv1_/conv2_/fc_ try to draw random numbers from it.
    std::mt19937 rng_;
    ConvLayer conv1_;
    ConvLayer conv2_;
    LinearLayer fc_;
};
