#pragma once

#include <vector>

// [channel][row][col]. Readability over raw performance here -- see README.md.
using Tensor3D = std::vector<std::vector<std::vector<float>>>;

Tensor3D make_tensor3d(size_t channels, size_t height, size_t width, float fill = 0.0f);

Tensor3D relu(Tensor3D x);
Tensor3D max_pool2d(const Tensor3D& x, int pool_size);
std::vector<float> flatten(const Tensor3D& x);
int argmax(const std::vector<float>& values);
