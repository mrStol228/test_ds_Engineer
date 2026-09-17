#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"
#include "csv.hpp"
#include "linear_regression.hpp"
#include "matrix.hpp"

namespace {

Matrix build_design_matrix(const std::vector<std::vector<double>>& raw_rows) {
    Matrix X(raw_rows.size(), DESIGN_COLUMN_COUNT);
    for (size_t i = 0; i < raw_rows.size(); ++i) {
        std::vector<double> engineered = engineer_features(raw_rows[i]);
        for (size_t j = 0; j < DESIGN_COLUMN_COUNT; ++j) {
            X(i, j) = engineered[j];
        }
    }
    return X;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string train_path = argc > 1 ? argv[1] : "../../regression/data/train.csv";
    std::string model_path = argc > 2 ? argv[2] : "models/weights.txt";

    std::cout << "Reading " << train_path << " ..." << std::endl;
    CsvTable table = read_csv(train_path);
    size_t n = table.rows.size();
    std::cout << "Loaded " << n << " rows, " << table.header.size() << " columns." << std::endl;

    std::vector<std::vector<double>> raw_features(n);
    std::vector<double> target(n);
    for (size_t i = 0; i < n; ++i) {
        const std::vector<double>& row = table.rows[i];
        raw_features[i].assign(row.begin(), row.begin() + RAW_FEATURE_COUNT);
        target[i] = row.back();  // last column is "target"
    }

    // 80/20 holdout split, shuffled with a fixed seed for reproducibility.
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t n_val = n / 5;
    size_t n_train = n - n_val;

    std::vector<std::vector<double>> train_raw(n_train), val_raw(n_val);
    std::vector<double> train_y(n_train), val_y(n_val);
    for (size_t i = 0; i < n_train; ++i) {
        train_raw[i] = raw_features[indices[i]];
        train_y[i] = target[indices[i]];
    }
    for (size_t i = 0; i < n_val; ++i) {
        val_raw[i] = raw_features[indices[n_train + i]];
        val_y[i] = target[indices[n_train + i]];
    }

    std::cout << "Fitting on " << n_train << " rows, holding out " << n_val << " for validation..." << std::endl;

    Matrix X_train = build_design_matrix(train_raw);
    std::vector<double> weights = fit_ols(X_train, train_y);

    Matrix X_val = build_design_matrix(val_raw);
    std::vector<std::vector<double>> val_design_rows(n_val);
    for (size_t i = 0; i < n_val; ++i) {
        val_design_rows[i].resize(DESIGN_COLUMN_COUNT);
        for (size_t j = 0; j < DESIGN_COLUMN_COUNT; ++j) {
            val_design_rows[i][j] = X_val(i, j);
        }
    }
    std::vector<double> val_pred = predict_all(weights, val_design_rows);
    std::cout << "Holdout RMSE (20% held out): " << rmse(val_y, val_pred) << std::endl;

    std::cout << "Refitting on all " << n << " rows..." << std::endl;
    Matrix X_full = build_design_matrix(raw_features);
    weights = fit_ols(X_full, target);

    std::filesystem::create_directories(std::filesystem::path(model_path).parent_path());
    save_weights(model_path, weights);
    std::cout << "Saved weights to " << model_path << std::endl;

    return 0;
}
