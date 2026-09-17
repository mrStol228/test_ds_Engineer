#include "cnn_model.hpp"

#include <stdexcept>

CnnModel::CnnModel() : rng_(42), conv1_(1, 8, rng_), conv2_(8, 16, rng_), fc_(16 * 7 * 7, 10, rng_) {}

void CnnModel::train(const std::vector<std::vector<double>>&, const std::vector<int>&) {
    throw std::logic_error("Training is out of scope for this exercise.");
}

int CnnModel::predict(const std::vector<double>& image) {
    if (image.size() != kInputLength) {
        throw std::invalid_argument("CnnModel::predict expects 784 values (a flattened 28x28 image)");
    }

    Tensor3D input = make_tensor3d(1, 28, 28);
    for (size_t r = 0; r < 28; ++r) {
        for (size_t c = 0; c < 28; ++c) {
            input[0][r][c] = static_cast<float>(image[r * 28 + c]);
        }
    }

    Tensor3D x = conv1_.forward(input);   // 1x28x28 -> 8x28x28
    x = relu(x);
    x = max_pool2d(x, 2);                 // -> 8x14x14
    x = conv2_.forward(x);                // -> 16x14x14
    x = relu(x);
    x = max_pool2d(x, 2);                 // -> 16x7x7

    std::vector<float> flat = flatten(x);       // -> 784
    std::vector<float> logits = fc_.forward(flat);  // -> 10

    return argmax(logits);
}
