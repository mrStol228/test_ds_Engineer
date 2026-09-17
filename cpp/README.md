# The same three tasks, in C++

A from-scratch C++ port of `islands/`, `regression/` and `mnist/` — written
as a learning project, not a requirement of the original task. No external
libraries: everything (matrix math, CSV parsing, a decision tree, a tiny
CNN forward pass) is implemented by hand with the C++ standard library, on
purpose, so there's nothing to install and nothing hidden behind a
library call.

Read them in this order — each one only explains what's *new* compared to
the last, so later ones assume you've read the earlier ones:

1. **`islands/`** — start here. No classes yet, just: compiled vs.
   interpreted, static types, `std::vector`, pass-by-value vs.
   pass-by-reference (the single biggest gotcha coming from Python),
   `std::queue`, and why out-of-bounds access doesn't raise a clean
   exception like Python's `IndexError`.
2. **`regression/`** — introduces classes, operator overloading, RAII
   (why nothing here calls `new`/`delete`), `const`-correctness, and does
   linear regression's normal equations (`w = (XᵀX)⁻¹Xᵀy`) with a
   hand-written matrix inverse (Gauss-Jordan elimination) instead of
   calling into NumPy/LAPACK.
3. **`mnist/`** — introduces polymorphism: abstract classes, `virtual`
   functions and dynamic dispatch, `std::unique_ptr` for owning
   polymorphic objects (and why you can't just store the base class by
   value), and a factory registry built from `std::function` lambdas
   (the C++ answer to Python's "just put the class in a dict"). Also
   includes a hand-rolled decision tree using `unique_ptr`-owned nodes —
   the natural first real use case for smart pointers.

Each subfolder builds independently with CMake:

```bash
cd cpp/<folder>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`build/` is git-ignored (regenerable from source, same reasoning as
`.venv/` for the Python projects) — always run the two commands above
after cloning, before anything will run.
