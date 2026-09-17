#pragma once

#include <random>
#include <vector>

#include "decision_tree.hpp"
#include "digit_classification_interface.hpp"

// Expects a flat vector of 784 values -- a 28x28 image flattened.
class RandomForestModel : public DigitClassificationInterface {
public:
    static constexpr size_t kInputLength = 28 * 28;

    RandomForestModel();

    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) override;
    int predict(const std::vector<double>& image) override;

private:
    void warm_start();

    std::mt19937 rng_;
    std::vector<DecisionTree> trees_;
};
