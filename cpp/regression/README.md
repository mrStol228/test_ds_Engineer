# Regression in C++ — classes, and doing your own linear algebra

Same task as `regression/` (Python): fit `target = feature_6² + feature_7`
from `train.csv` and predict `hidden_test.csv`, scored by RMSE. This file
assumes you've read `cpp/islands/README.md` first (types, compilation,
value vs. reference, `std::vector`) — here we cover what's *new*: classes,
and reimplementing what pandas/numpy/scikit-learn normally do for you.

## 1. Why this needs a class, and `islands/` didn't

`islands/` only needed plain functions operating on a `Grid` (a type alias
for a `vector<vector<int>>`). A matrix for linear algebra needs more: it
needs to carry its own *behavior* — multiply itself by another matrix,
transpose itself, invert itself — and it needs to guarantee its internal
data can never end up in an inconsistent state from outside code. That's
what a C++ **class** buys you (`matrix.hpp`):

```cpp
class Matrix {
public:
    Matrix(size_t rows, size_t cols, double fill = 0.0);
    double& operator()(size_t r, size_t c);
    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;
    Matrix inverse() const;
private:
    size_t rows_, cols_;
    std::vector<double> data_;
};
```

`public:` members are the class's interface — what code outside the class
is allowed to touch. `private:` members (`rows_`, `cols_`, `data_`) are
invisible from outside; only `Matrix`'s own methods can see them. This is
**encapsulation**: nothing outside `Matrix` can accidentally set `rows_`
to one value while `data_` is still sized for a different one, because
nothing outside can touch `rows_` at all. Python has no real equivalent —
`self._data` on a Python object is a convention (the underscore *asks*
callers not to touch it), not an enforced rule; in C++, `private` is
enforced by the compiler.

## 2. `operator()` overloading — making `A(i, j)` mean something

```cpp
double& Matrix::operator()(size_t r, size_t c) {
    return data_[r * cols_ + c];
}
```

C++ lets you redefine what an operator (`()`, `+`, `==`, ...) does for
your own type. This method is what makes `A(2, 3) = 5.0;` valid code —
without it, `A(2, 3)` would be meaningless, since `Matrix` isn't a
function. Python's equivalent trick is dunder methods (`__call__`,
`__add__`) — same idea, different spelling.

Notice the return type is `double&` — a *reference* to the double inside
the matrix, not a copy (see `cpp/islands/README.md` section 4 on
references). That's what makes `A(2, 3) = 5.0;` able to actually reach in
and change the stored value, instead of setting a copy that's immediately
thrown away. `operator()` also has a `const` overload right below it in
`matrix.hpp`, returning a plain `double` (no `&`) — used when you only
want to *read* a value from a matrix you're not allowed to modify (see
section 6 on `const`).

## 3. One flat `vector<double>`, not a `vector<vector<double>>`

`islands/` stores its grid as `vector<vector<int>>` — a vector of vectors,
one small heap allocation per row. `Matrix` instead stores everything in
a *single* `std::vector<double>` and computes 2-D positions by hand:

```cpp
data_[r * cols_ + c]   // row r, column c, in one flat buffer
```

This is **row-major layout**: row 0's values, then row 1's values, and so
on, back to back in memory. It matters for two reasons: one big
allocation instead of `rows` separate ones (faster to create, and the
data ends up contiguous in memory, which is much friendlier to the CPU
cache during the multiply loops in section 5) — and it sidesteps a subtle
bug class where a `vector<vector<double>>`'s rows could technically end
up with different lengths, which would silently corrupt a "matrix" that's
supposed to be rectangular. `Grid` in `islands/` didn't need this — grids
there are small and read once, so the simplicity of nested vectors won.

## 4. Nobody calls `new` or `delete` here — and that's the point

If you've heard C++ has a reputation for manual memory management and
crash-prone pointers: notice that neither `Matrix` nor anything else in
this project calls `new`, `delete`, `malloc`, or `free`, anywhere.
`std::vector<double> data_` owns a heap buffer internally, and — this is
the important part — **it frees that buffer automatically** the moment
the `Matrix` that contains it goes out of scope, with no code from us
required. This pattern (a value's constructor acquires a resource, its
destructor releases it automatically) is called **RAII** and it's the
default, idiomatic way to manage memory in modern C++. The `new`/`delete`
style you may have heard about is the old, manual way, mostly relegated
to either implementing a container *like* `vector` from scratch, or
legacy code — application code like this project practically never needs
it.

## 5. The actual math: normal equations, by hand

Linear regression by ordinary least squares finds the weight vector `w`
minimizing `‖y - Xw‖²`. The closed-form solution (derived in full, with
the calculus, in the `regression/` Python project's accompanying
explanations) is:

```
w = (XᵀX)⁻¹ Xᵀy
```

`linear_regression.cpp`'s `fit_ols` is a direct, literal translation of
that formula into matrix calls:

```cpp
Matrix Xt = X.transpose();
Matrix XtX = Xt.multiply(X);
Matrix XtY = Xt.multiply(Y);
Matrix XtX_inv = XtX.inverse();
Matrix W = XtX_inv.multiply(XtY);
```

Every symbol in the math formula has a matching line of code. This is
exactly what `numpy`/`scikit-learn` do under the hood in the Python
version too — the difference is that there, `LinearRegression().fit(...)`
calls into LAPACK (a decades-old, heavily optimized Fortran/C linear
algebra library) to do this; here, `Matrix::inverse()` does it itself,
in about 40 lines, so you can see every step.

### How `Matrix::inverse()` actually works

It implements **Gauss-Jordan elimination**: start with `[A | I]` (the
matrix next to an identity matrix of the same size), and repeatedly apply
row operations — swap rows, scale a row, subtract a multiple of one row
from another — until the left side becomes `I`. Whatever the right side
has turned into by then is `A⁻¹`. The code does this column by column
(each column is one "pivot"):

1. **Partial pivoting**: before eliminating column `pivot`, scan that
   column for the row with the largest absolute value and swap it into
   the pivot position. Skipping this and just using whatever's already
   there works *mathematically*, but can be numerically disastrous —
   dividing by a tiny pivot value amplifies floating-point rounding error
   enormously. This single check is most of what separates a
   "textbook-correct" inverse from one that's actually usable on real
   data.
2. **Normalize the pivot row** so the pivot position becomes exactly `1`.
3. **Eliminate that column from every other row** by subtracting the
   right multiple of the pivot row, driving every other entry in that
   column to `0`.

Do this for every column and the left side has become the identity — and
the right side, which started as the identity and had every operation
applied to it too, has become the inverse.

If a pivot value is essentially zero even after picking the best row
(`best_value < 1e-12`), the matrix is **singular** (not invertible) —
`Matrix::inverse()` throws `std::runtime_error` rather than dividing by
(near-)zero and returning garbage. This can't actually happen for the
regression problem here — see section 7 — but the check exists because a
matrix library that silently returns nonsense on bad input is worse than
one that stops.

## 6. `const` — "I promise not to change this"

You'll see `const Matrix& other` and `Matrix multiply(...) const` all
over `matrix.hpp`. Two different uses of the same keyword:

- `const Matrix& other` (in a parameter list) — pass `other` by reference
  (no copy, see the islands README), *and* promise the function won't
  modify it. The compiler enforces the promise: try to write to `other`
  inside a function that took it as `const Matrix&` and it won't compile.
- `Matrix multiply(const Matrix& other) const` — that trailing `const`
  applies to `*this`, i.e. it promises the method won't modify the
  `Matrix` it's called on either. `A.multiply(B)` doesn't change `A`; it
  returns a brand new `Matrix` with the result.

Python has no equivalent — nothing stops a Python method from mutating
`self` or its arguments, it's purely a documentation convention there
(and even that convention is easy to violate accidentally). In C++, if a
method is marked `const`, the compiler physically will not let it modify
member data, full stop.

## 7. The "bias trick" — why there's a column of `1.0`s

A line predicting `y = w·x + b` has two things to learn: the weight `w`
*and* the intercept `b`. Rather than special-casing `b` everywhere,
`engineer_features` (`common.cpp`) appends a constant `1.0` as the last
column of every row:

```cpp
row.push_back(1.0);  // bias term
```

Multiplying that column by its corresponding weight just gives back that
weight itself (`w_bias * 1.0 == w_bias`) — so the intercept becomes
*indistinguishable*, mathematically, from any other feature's weight. One
formula (`w = (XᵀX)⁻¹Xᵀy`) then learns the slope on every real feature
*and* the intercept in one shot, with no special case anywhere in
`Matrix` or `fit_ols`. `engineer_features` builds `DESIGN_COLUMN_COUNT =
53 + 1 + 1 = 55` columns per row this way: the 53 raw features, the
engineered `feature_6²`, and this bias column last.

Since `train = feature_6² + feature_7` exactly (see the Python EDA), the
"ideal" weight vector is: `1.0` for the engineered `feature_6²` column,
`1.0` for raw `feature_7`, `0.0` for every other raw feature, and `0.0`
for the bias. That's almost exactly what `fit_ols` recovers on the real
90,000-row dataset — see section 9.

## 8. Reading a CSV without pandas

`csv.cpp`'s `read_csv` is about 20 lines: read the header line, then for
every remaining line, split on commas and parse each piece as a `double`
with `std::stod`. This is **not** a general-purpose CSV parser — it
doesn't handle quoted fields, embedded commas, or escaped characters —
and it doesn't need to, because `train.csv`/`hidden_test.csv` are
machine-generated, plain numeric CSVs with none of those complications.
Pandas' `read_csv` handles all of that generality (and a hundred other
edge cases) because it has to work on *any* CSV a human hands it; this
one only has to work on this one.

## 9. Saving the model: a text file instead of pickling

Python's `joblib.dump` (used in `regression/train.py`) serializes the
*entire* fitted scikit-learn object — its Python class, its internal
NumPy arrays, everything — using `pickle` under the hood, so
`joblib.load` can reconstruct an object you can call `.predict()` on
directly. That machinery doesn't exist here (and doesn't need to): the
*only* thing a fitted `LinearRegression` actually needs to make
predictions is its 55 numbers. `save_weights`/`load_weights` write and
read exactly that — one `double` per line, full precision
(`file.precision(17)`, enough to round-trip a `double` exactly) — as
plain text you could open and read yourself. `models/weights.txt` in this
folder *is* the trained model.

## 10. Running it on the real data

```bash
cd cpp/regression
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/test_regression                        # unit tests
./build/train                                    # trains on regression/data/train.csv
./build/predict                                  # predicts regression/data/hidden_test.csv
```

By default `train`/`predict` read from `../../regression/data/` — the
same `train.csv`/`hidden_test.csv` the Python project uses, so the data
isn't duplicated on disk. Pass explicit paths as arguments if you've put
the CSVs somewhere else: `./build/train path/to/train.csv models/weights.txt`.

On the real 90,000-row dataset: reading the CSV, building a 90000x55
design matrix, inverting a 55x55 matrix *twice* (once on the 72,000-row
training split, once again on the full data) and writing the result out
takes about **2.5 seconds** — including the disk I/O. Holdout RMSE comes
out around `1e-11` — same story as the Python version's `≈ 0`, just a
slightly different tiny number because this is a hand-rolled Gauss-Jordan
in plain `double`s rather than LAPACK's more numerically careful
routines, and because the 80/20 split lands on different rows (different
random number generator, different seed semantics).

## 11. What's deliberately left out, vs. the Python version

- **No cross-validation** — just one 80/20 holdout split, not 5-fold CV.
  Since this model is an exact closed-form fit (not something stochastic
  like a Random Forest), a single split is enough to demonstrate it
  generalizes; 5-fold CV would mean writing a fold-splitting loop for
  very little additional insight here.
- **No Random Forest / LightGBM fallback.** The Python version keeps
  those as a hedge in case the hidden formula isn't real — implementing a
  decision tree from scratch is a reasonable *next* C++ exercise, but a
  separate one from "learn matrices and classes."
- **The CSV parser is intentionally narrow** (section 8) — it would need
  to grow real quoting/escaping support to be safe on arbitrary CSVs.

## 12. Side-by-side with the Python version

| | Python (`regression/`) | C++ (`cpp/regression/`) |
|---|---|---|
| Read CSV | `pandas.read_csv` | hand-rolled `read_csv` (numeric-only) |
| Matrix / array | `numpy.ndarray` | hand-rolled `Matrix` class |
| Fit OLS | `sklearn.LinearRegression().fit(...)` (LAPACK inside) | `fit_ols` calling `Matrix::inverse()` (Gauss-Jordan) |
| Save model | `joblib.dump` (pickles the whole object) | `save_weights` (55 numbers, one per line) |
| Load model | `joblib.load` | `load_weights` |
| Bad input | raises a Python exception | `throw std::invalid_argument` / `std::runtime_error` |
| Encapsulation | `self._x` (convention only) | `private:` (compiler-enforced) |
