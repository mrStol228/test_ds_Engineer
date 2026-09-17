#pragma once

#include <memory>
#include <string>
#include <vector>

#include "digit_classification_interface.hpp"

// Uniform entry point: predict() always takes a flat 784-value vector (a
// 28x28 image, row-major) and always returns an int 0-9, no matter which
// algorithm is selected.
class DigitClassifier {
public:
    static constexpr size_t kCanonicalLength = 28 * 28;

    explicit DigitClassifier(const std::string& algorithm);

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    int predict(const std::vector<double>& canonical_image);

private:
    std::vector<double> adapt(const std::vector<double>& canonical_image) const;

    std::string algorithm_;
    std::unique_ptr<DigitClassificationInterface> model_;
};
