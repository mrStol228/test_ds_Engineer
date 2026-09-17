#include "linear_regression.hpp"

#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

std::vector<double> fit_ols(const Matrix& X, const std::vector<double>& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("fit_ols: X and y have different row counts");
    }

    Matrix Xt = X.transpose();
    Matrix XtX = Xt.multiply(X);

    Matrix Y(y.size(), 1);
    for (size_t i = 0; i < y.size(); ++i) {
        Y(i, 0) = y[i];
    }
    Matrix XtY = Xt.multiply(Y);

    Matrix XtX_inv = XtX.inverse();
    Matrix W = XtX_inv.multiply(XtY);

    std::vector<double> weights(W.rows());
    for (size_t i = 0; i < W.rows(); ++i) {
        weights[i] = W(i, 0);
    }
    return weights;
}

double predict_row(const std::vector<double>& weights, const std::vector<double>& design_row) {
    if (weights.size() != design_row.size()) {
        throw std::invalid_argument("predict_row: weights and design_row have different lengths");
    }
    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * design_row[i];
    }
    return sum;
}

std::vector<double> predict_all(const std::vector<double>& weights, const std::vector<std::vector<double>>& design_rows) {
    std::vector<double> predictions;
    predictions.reserve(design_rows.size());
    for (const auto& row : design_rows) {
        predictions.push_back(predict_row(weights, row));
    }
    return predictions;
}

double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("rmse: vectors have different lengths");
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / static_cast<double>(y_true.size()));
}

void save_weights(const std::string& path, const std::vector<double>& weights) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("save_weights: could not open " + path + " for writing");
    }
    file.precision(17);
    for (double w : weights) {
        file << w << '\n';
    }
}

std::vector<double> load_weights(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("load_weights: could not open " + path);
    }
    std::vector<double> weights;
    double value;
    while (file >> value) {
        weights.push_back(value);
    }
    return weights;
}
