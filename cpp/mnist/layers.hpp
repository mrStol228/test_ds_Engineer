#pragma once

#include <random>
#include <vector>

#include "tensor3d.hpp"

// A 3x3, stride-1, "same"-padding convolution. Weights are randomly
// initialized at construction and never trained -- see README.md.
class ConvLayer {
public:
    ConvLayer(int in_channels, int out_channels, std::mt19937& rng);
    Tensor3D forward(const Tensor3D& input) const;

private:
    int in_channels_;
    int out_channels_;
    static constexpr int kKernelSize = 3;
    // weights_[out][in][kr][kc], bias_[out]
    std::vector<std::vector<std::vector<std::vector<float>>>> weights_;
    std::vector<float> bias_;
};

class LinearLayer {
public:
    LinearLayer(int in_features, int out_features, std::mt19937& rng);
    std::vector<float> forward(const std::vector<float>& input) const;

private:
    int in_features_;
    int out_features_;
    std::vector<std::vector<float>> weights_;  // [out][in]
    std::vector<float> bias_;                   // [out]
};
