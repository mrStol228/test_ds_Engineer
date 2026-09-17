#pragma once

#include <string>
#include <vector>

#include "matrix.hpp"

// Ordinary least squares via the normal equations: w = (XtX)^-1 Xt y.
// X is n_samples x n_features (n_features == DESIGN_COLUMN_COUNT, bias column included).
std::vector<double> fit_ols(const Matrix& X, const std::vector<double>& y);

double predict_row(const std::vector<double>& weights, const std::vector<double>& design_row);

std::vector<double> predict_all(const std::vector<double>& weights, const std::vector<std::vector<double>>& design_rows);

double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred);

void save_weights(const std::string& path, const std::vector<double>& weights);
std::vector<double> load_weights(const std::string& path);
