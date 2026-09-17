#include "common.hpp"

#include <stdexcept>

std::vector<double> engineer_features(const std::vector<double>& raw_features) {
    if (raw_features.size() != RAW_FEATURE_COUNT) {
        throw std::invalid_argument("engineer_features: expected 53 raw feature values");
    }

    std::vector<double> row;
    row.reserve(DESIGN_COLUMN_COUNT);

    for (double value : raw_features) {
        row.push_back(value);
    }

    double feature_6 = raw_features[QUADRATIC_SOURCE_INDEX];
    row.push_back(feature_6 * feature_6);

    row.push_back(1.0);  // bias term

    return row;
}
