#include <cmath>
#include <iostream>

#include "common.hpp"
#include "linear_regression.hpp"
#include "matrix.hpp"

namespace {

int failures = 0;

void check(const std::string& name, bool ok) {
    if (ok) {
        std::cout << "PASS  " << name << std::endl;
    } else {
        std::cout << "FAIL  " << name << std::endl;
        ++failures;
    }
}

bool close(double a, double b, double tol = 1e-8) {
    return std::abs(a - b) < tol;
}

}  // namespace

int main() {
    // Matrix::multiply on a known 2x2 case.
    Matrix a(2, 2);
    a(0, 0) = 1; a(0, 1) = 2;
    a(1, 0) = 3; a(1, 1) = 4;
    Matrix identity = Matrix::identity(2);
    Matrix product = a.multiply(identity);
    check("matrix_multiply_by_identity", close(product(0, 0), 1) && close(product(1, 1), 4));

    // Matrix::inverse: A * A^-1 should be the identity.
    Matrix inv = a.inverse();
    Matrix should_be_identity = a.multiply(inv);
    check("matrix_inverse_roundtrip",
          close(should_be_identity(0, 0), 1) && close(should_be_identity(0, 1), 0) &&
          close(should_be_identity(1, 0), 0) && close(should_be_identity(1, 1), 1));

    // engineer_features: 53 raw values in, 55 out, feature 6 squared in the right slot.
    std::vector<double> raw(RAW_FEATURE_COUNT, 0.0);
    raw[QUADRATIC_SOURCE_INDEX] = 3.0;
    std::vector<double> engineered = engineer_features(raw);
    check("engineer_features_length", engineered.size() == DESIGN_COLUMN_COUNT);
    check("engineer_features_squares_feature_6", close(engineered[RAW_FEATURE_COUNT], 9.0));
    check("engineer_features_bias_term", close(engineered.back(), 1.0));

    // fit_ols on a tiny synthetic problem: y = 2*x + 3, exactly, no noise.
    // Design matrix columns: [x, bias].
    Matrix X(4, 2);
    std::vector<double> y = {5.0, 7.0, 9.0, 11.0};  // x = 1, 2, 3, 4
    for (int i = 0; i < 4; ++i) {
        X(i, 0) = i + 1;
        X(i, 1) = 1.0;
    }
    std::vector<double> weights = fit_ols(X, y);
    check("fit_ols_recovers_slope", close(weights[0], 2.0));
    check("fit_ols_recovers_intercept", close(weights[1], 3.0));

    // rmse of a perfect prediction is 0.
    check("rmse_perfect_prediction", close(rmse(y, y), 0.0));

    if (failures == 0) {
        std::cout << "\nAll tests passed." << std::endl;
        return 0;
    }
    std::cout << "\n" << failures << " test(s) failed." << std::endl;
    return 1;
}
