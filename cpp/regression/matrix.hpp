#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix(size_t rows, size_t cols, double fill = 0.0);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    double& operator()(size_t r, size_t c);
    double operator()(size_t r, size_t c) const;

    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;
    Matrix inverse() const;

    static Matrix identity(size_t n);

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
};
