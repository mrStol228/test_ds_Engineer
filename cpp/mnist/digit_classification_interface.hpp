#pragma once

#include <vector>

// Every concrete model validates and interprets the length of `image` itself
// (784 for CnnModel/RandomForestModel, 100 for RandomModel) -- see README.md,
// section "why predict() takes a flat vector<double> here".
class DigitClassificationInterface {
public:
    virtual ~DigitClassificationInterface() = default;

    virtual void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) = 0;
    virtual int predict(const std::vector<double>& image) = 0;
};
