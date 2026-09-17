#include "tensor3d.hpp"

#include <algorithm>
#include <stdexcept>

Tensor3D make_tensor3d(size_t channels, size_t height, size_t width, float fill) {
    return Tensor3D(channels, std::vector<std::vector<float>>(height, std::vector<float>(width, fill)));
}

Tensor3D relu(Tensor3D x) {
    for (auto& channel : x) {
        for (auto& row : channel) {
            for (float& value : row) {
                value = std::max(0.0f, value);
            }
        }
    }
    return x;
}

Tensor3D max_pool2d(const Tensor3D& x, int pool_size) {
    size_t channels = x.size();
    size_t height = x[0].size();
    size_t width = x[0][0].size();
    size_t out_h = height / pool_size;
    size_t out_w = width / pool_size;

    Tensor3D result = make_tensor3d(channels, out_h, out_w);
    for (size_t ch = 0; ch < channels; ++ch) {
        for (size_t r = 0; r < out_h; ++r) {
            for (size_t c = 0; c < out_w; ++c) {
                float best = x[ch][r * pool_size][c * pool_size];
                for (int dr = 0; dr < pool_size; ++dr) {
                    for (int dc = 0; dc < pool_size; ++dc) {
                        best = std::max(best, x[ch][r * pool_size + dr][c * pool_size + dc]);
                    }
                }
                result[ch][r][c] = best;
            }
        }
    }
    return result;
}

std::vector<float> flatten(const Tensor3D& x) {
    std::vector<float> result;
    for (const auto& channel : x) {
        for (const auto& row : channel) {
            for (float value : row) {
                result.push_back(value);
            }
        }
    }
    return result;
}

int argmax(const std::vector<float>& values) {
    if (values.empty()) {
        throw std::invalid_argument("argmax: empty vector");
    }
    size_t best_index = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best_index]) {
            best_index = i;
        }
    }
    return static_cast<int>(best_index);
}
