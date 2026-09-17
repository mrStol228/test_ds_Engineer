#include "random_model.hpp"

#include <stdexcept>

RandomModel::RandomModel(unsigned seed) : rng_(seed), dist_(0, 9) {}

void RandomModel::train(const std::vector<std::vector<double>>&, const std::vector<int>&) {
    throw std::logic_error("Training is out of scope for this exercise.");
}

int RandomModel::predict(const std::vector<double>& image) {
    if (image.size() != kInputLength) {
        throw std::invalid_argument("RandomModel::predict expects 100 values (a 10x10 crop)");
    }
    return dist_(rng_);
}
