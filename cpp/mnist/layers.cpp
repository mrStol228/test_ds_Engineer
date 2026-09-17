#include "layers.hpp"

#include <cmath>
#include <stdexcept>

ConvLayer::ConvLayer(int in_channels, int out_channels, std::mt19937& rng)
    : in_channels_(in_channels), out_channels_(out_channels) {
    // Roughly matches how PyTorch initializes conv weights: uniform in
    // +/- 1/sqrt(fan_in), where fan_in = in_channels * kernel_size^2.
    float fan_in = static_cast<float>(in_channels_ * kKernelSize * kKernelSize);
    float bound = 1.0f / std::sqrt(fan_in);
    std::uniform_real_distribution<float> dist(-bound, bound);

    weights_.assign(out_channels_, std::vector<std::vector<std::vector<float>>>(
                                        in_channels_, std::vector<std::vector<float>>(
                                                          kKernelSize, std::vector<float>(kKernelSize))));
    for (auto& per_out : weights_) {
        for (auto& per_in : per_out) {
            for (auto& row : per_in) {
                for (float& w : row) {
                    w = dist(rng);
                }
            }
        }
    }

    bias_.resize(out_channels_);
    for (float& b : bias_) {
        b = dist(rng);
    }
}

Tensor3D ConvLayer::forward(const Tensor3D& input) const {
    size_t height = input[0].size();
    size_t width = input[0][0].size();
    int pad = kKernelSize / 2;  // padding=1 for a 3x3 kernel keeps height/width unchanged

    Tensor3D output = make_tensor3d(out_channels_, height, width);

    for (int oc = 0; oc < out_channels_; ++oc) {
        for (size_t r = 0; r < height; ++r) {
            for (size_t c = 0; c < width; ++c) {
                float sum = bias_[oc];
                for (int ic = 0; ic < in_channels_; ++ic) {
                    for (int kr = 0; kr < kKernelSize; ++kr) {
                        for (int kc = 0; kc < kKernelSize; ++kc) {
                            int ir = static_cast<int>(r) + kr - pad;
                            int ic_pos = static_cast<int>(c) + kc - pad;
                            if (ir < 0 || ir >= static_cast<int>(height) || ic_pos < 0 ||
                                ic_pos >= static_cast<int>(width)) {
                                continue;  // implicit zero-padding
                            }
                            sum += weights_[oc][ic][kr][kc] * input[ic][ir][ic_pos];
                        }
                    }
                }
                output[oc][r][c] = sum;
            }
        }
    }
    return output;
}

LinearLayer::LinearLayer(int in_features, int out_features, std::mt19937& rng)
    : in_features_(in_features), out_features_(out_features) {
    float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
    std::uniform_real_distribution<float> dist(-bound, bound);

    weights_.assign(out_features_, std::vector<float>(in_features_));
    for (auto& row : weights_) {
        for (float& w : row) {
            w = dist(rng);
        }
    }

    bias_.resize(out_features_);
    for (float& b : bias_) {
        b = dist(rng);
    }
}

std::vector<float> LinearLayer::forward(const std::vector<float>& input) const {
    if (static_cast<int>(input.size()) != in_features_) {
        throw std::invalid_argument("LinearLayer::forward: unexpected input size");
    }
    std::vector<float> output(out_features_);
    for (int o = 0; o < out_features_; ++o) {
        float sum = bias_[o];
        for (int i = 0; i < in_features_; ++i) {
            sum += weights_[o][i] * input[i];
        }
        output[o] = sum;
    }
    return output;
}
