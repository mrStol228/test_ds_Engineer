#include "digit_classifier.hpp"

#include <functional>
#include <map>
#include <stdexcept>

#include "cnn_model.hpp"
#include "random_forest_model.hpp"
#include "random_model.hpp"

namespace {

using Factory = std::function<std::unique_ptr<DigitClassificationInterface>()>;

const std::map<std::string, Factory>& registry() {
    static const std::map<std::string, Factory> table = {
        {"cnn", [] { return std::make_unique<CnnModel>(); }},
        {"rf", [] { return std::make_unique<RandomForestModel>(); }},
        {"rand", [] { return std::make_unique<RandomModel>(); }},
    };
    return table;
}

constexpr size_t kCanonicalSide = 28;

std::vector<double> center_crop(const std::vector<double>& image, int crop_size) {
    int top = (static_cast<int>(kCanonicalSide) - crop_size) / 2;
    std::vector<double> out;
    out.reserve(static_cast<size_t>(crop_size) * crop_size);
    for (int r = 0; r < crop_size; ++r) {
        for (int c = 0; c < crop_size; ++c) {
            out.push_back(image[(top + r) * kCanonicalSide + (top + c)]);
        }
    }
    return out;
}

}  // namespace

DigitClassifier::DigitClassifier(const std::string& algorithm) : algorithm_(algorithm) {
    const auto& table = registry();
    auto it = table.find(algorithm);
    if (it == table.end()) {
        throw std::invalid_argument("Unknown algorithm: " + algorithm);
    }
    model_ = it->second();  // calls the factory lambda -> a fresh CnnModel/RandomForestModel/RandomModel
}

void DigitClassifier::train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    model_->train(X, y);
}

int DigitClassifier::predict(const std::vector<double>& canonical_image) {
    if (canonical_image.size() != kCanonicalLength) {
        throw std::invalid_argument("DigitClassifier::predict expects 784 values (a flattened 28x28 image)");
    }
    return model_->predict(adapt(canonical_image));
}

std::vector<double> DigitClassifier::adapt(const std::vector<double>& canonical_image) const {
    if (algorithm_ == "cnn" || algorithm_ == "rf") {
        return canonical_image;
    }
    if (algorithm_ == "rand") {
        return center_crop(canonical_image, 10);
    }
    throw std::logic_error("unreachable algorithm: " + algorithm_);
}
