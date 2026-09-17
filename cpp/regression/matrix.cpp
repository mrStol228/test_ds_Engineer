#include "matrix.hpp"

#include <cmath>
#include <stdexcept>

Matrix::Matrix(size_t rows, size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

double& Matrix::operator()(size_t r, size_t c) {
    return data_[r * cols_ + c];
}

double Matrix::operator()(size_t r, size_t c) const {
    return data_[r * cols_ + c];
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < cols_; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix::multiply: inner dimensions do not match");
    }

    Matrix result(rows_, other.cols_);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t k = 0; k < cols_; ++k) {
            double a_rk = (*this)(r, k);
            if (a_rk == 0.0) {
                continue;
            }
            for (size_t c = 0; c < other.cols_; ++c) {
                result(r, c) += a_rk * other(k, c);
            }
        }
    }
    return result;
}

Matrix Matrix::identity(size_t n) {
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix::inverse: matrix must be square");
    }

    size_t n = rows_;
    Matrix work = *this;
    Matrix inv = Matrix::identity(n);

    for (size_t pivot = 0; pivot < n; ++pivot) {
        size_t best_row = pivot;
        double best_value = std::abs(work(pivot, pivot));
        for (size_t r = pivot + 1; r < n; ++r) {
            double value = std::abs(work(r, pivot));
            if (value > best_value) {
                best_value = value;
                best_row = r;
            }
        }

        if (best_value < 1e-12) {
            throw std::runtime_error("Matrix::inverse: matrix is singular (or numerically too close to it)");
        }

        if (best_row != pivot) {
            for (size_t c = 0; c < n; ++c) {
                std::swap(work(pivot, c), work(best_row, c));
                std::swap(inv(pivot, c), inv(best_row, c));
            }
        }

        double pivot_value = work(pivot, pivot);
        for (size_t c = 0; c < n; ++c) {
            work(pivot, c) /= pivot_value;
            inv(pivot, c) /= pivot_value;
        }

        for (size_t r = 0; r < n; ++r) {
            if (r == pivot) {
                continue;
            }
            double factor = work(r, pivot);
            if (factor == 0.0) {
                continue;
            }
            for (size_t c = 0; c < n; ++c) {
                work(r, c) -= factor * work(pivot, c);
                inv(r, c) -= factor * inv(pivot, c);
            }
        }
    }

    return inv;
}
