#pragma once

#include <cstddef>
#include <vector>

// train.csv / hidden_test.csv columns "0".."52" -- 53 raw features.
constexpr size_t RAW_FEATURE_COUNT = 53;

// Column "6" is the one EDA found has a quadratic relationship with target.
constexpr size_t QUADRATIC_SOURCE_INDEX = 6;

// RAW_FEATURE_COUNT raw features + 1 engineered (feature_6 squared) + 1 bias term.
constexpr size_t DESIGN_COLUMN_COUNT = RAW_FEATURE_COUNT + 2;

// Turns 53 raw feature values into the DESIGN_COLUMN_COUNT-long row used by the
// model: the 53 raw values, then feature_6 squared, then a constant 1.0 (the
// bias/intercept term -- see README.md, section "the bias trick").
std::vector<double> engineer_features(const std::vector<double>& raw_features);
