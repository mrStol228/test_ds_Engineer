# MNIST Classifier in C++ — interfaces, virtual functions, and owning pointers

Same task as `mnist/` (Python): one `DigitClassifier` in front of three
unrelated algorithms (CNN, "Random Forest", random baseline), each
expecting different input internally. Read `cpp/islands/README.md` and
`cpp/regression/README.md` first — this file only covers what's new:
**polymorphism**, and the pointer machinery C++ needs to make it work.

If you read the Python explanation for this same task (the one covering
`DigitClassificationInterface`, `ABC`, Strategy/Adapter), keep it open —
this is the *exact same design*, reimplemented. Every section below names
its Python equivalent.

## 1. Abstract classes: C++'s version of Python's `ABC`

Python's `DigitClassificationInterface(ABC)` uses `@abstractmethod` to
force every subclass to implement `train` and `predict`, checked when you
try to construct the object. C++ has a direct equivalent —
`digit_classification_interface.hpp`:

```cpp
class DigitClassificationInterface {
public:
    virtual ~DigitClassificationInterface() = default;
    virtual void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y) = 0;
    virtual int predict(const std::vector<double>& image) = 0;
};
```

`virtual ... = 0;` is a **pure virtual function** — it has no body, and it
makes the *whole class* abstract: exactly like Python's `ABC`,
`DigitClassificationInterface` cannot be instantiated
(`DigitClassificationInterface x;` is a compile error), and neither can
any subclass that forgets to implement one of these two methods. This is
the same guarantee as Python's `@abstractmethod`, just caught by the
compiler instead of at object-construction time.

`virtual ~DigitClassificationInterface() = default;` is new, though:
a **virtual destructor**. Skip it, and destroying a `CnnModel` through a
`DigitClassificationInterface*` pointer (which is exactly what
`DigitClassifier` does) would only run `DigitClassificationInterface`'s
destructor, not `CnnModel`'s — silently leaking anything `CnnModel` owns.
Python doesn't have this trap because its garbage collector doesn't care
what "declared type" a variable has when deciding what to clean up; C++
does, unless you mark the destructor `virtual`. Rule of thumb: **any
class meant to be inherited from and used through a base-class pointer
needs a virtual destructor.**

## 2. `virtual` and dynamic dispatch — how one call reaches three different bodies

```cpp
class CnnModel : public DigitClassificationInterface {
    int predict(const std::vector<double>& image) override { ... }
};
```

`class CnnModel : public DigitClassificationInterface` is inheritance —
same as Python's `class CnnModel(DigitClassificationInterface):`.
`override` isn't required by the compiler, but you should always write
it: it tells the compiler "I intend this to replace a virtual function
from the base class" — if you typo the signature so it *doesn't* actually
override anything, `override` turns that from a silent bug into a
compile error.

Here's the part with no Python equivalent to point at, because Python
*always* works this way and doesn't need a keyword for it: when you call
`model->predict(x)` through a `DigitClassificationInterface*`, C++ has to
decide, **at runtime**, whether that pointer actually points to a
`CnnModel`, a `RandomForestModel`, or a `RandomModel` — and jump to the
correct one's `predict`. It does this through a hidden lookup table (a
"vtable") that every polymorphic object carries a pointer to. `virtual`
is what turns that runtime lookup on; without it, C++ would resolve which
function to call at *compile* time, based only on the pointer's declared
type (`DigitClassificationInterface*`) — always calling the base class's
version, no matter what it actually points to. Python method calls are
*always* resolved dynamically like this (that's what "everything is an
object with its own `__class__`" gets you for free); C++ makes you opt in
with `virtual`, in exchange for not paying that lookup cost on ordinary,
non-polymorphic function calls.

## 3. `std::unique_ptr<Base>` — why this can't just be a `DigitClassificationInterface`

`DigitClassifier` (`digit_classifier.hpp`) stores its model like this:

```cpp
std::unique_ptr<DigitClassificationInterface> model_;
```

Not `DigitClassificationInterface model_;`. Two reasons, both fundamental:

1. **You can't have a variable of an abstract type at all** — there's no
   such thing as "just a `DigitClassificationInterface`, no more, no
   less"; it only exists as a `CnnModel` or a `RandomForestModel` or a
   `RandomModel`, i.e. through a pointer or reference to one of those.
2. Even for a non-abstract base class, storing a `CnnModel` in a plain
   `DigitClassificationInterface model_;` field would silently
   **truncate** it down to just the base-class part — the CNN's actual
   weights and layers would be sliced off ("object slicing"). Polymorphism
   in C++ only works through pointers or references.

`std::unique_ptr<T>` is a **smart pointer**: it owns a `T` allocated on
the heap, and — same RAII idea as `std::vector` in `cpp/regression/` —
automatically calls `delete` on it when the `unique_ptr` itself is
destroyed. `std::make_unique<CnnModel>()` allocates a `CnnModel` and
wraps it in a `unique_ptr<CnnModel>`, which converts implicitly to
`unique_ptr<DigitClassificationInterface>` because `CnnModel` *is a*
`DigitClassificationInterface`. This is the direct, safe C++ replacement
for what in Python is just `self._model = model_cls()` — Python's garbage
collector frees the object whenever nothing references it anymore;
`unique_ptr` gets you the same "I don't have to remember to free this"
guarantee, deterministically, the moment it goes out of scope, with zero
runtime overhead beyond a single pointer's worth of memory.

## 4. The factory registry: `std::map<std::string, std::function<...>>`

Python's version:

```python
_ALGORITHMS = {"cnn": CnnModel, "rf": RandomForestModel, "rand": RandomModel}
self._model = _ALGORITHMS[algorithm]()
```

works because in Python, a class *is* a value — `CnnModel` itself can be
put straight into a dict and called later. C++ classes are not
first-class values like that; you can't put "the type `CnnModel`" into a
`std::map`. What you *can* put in a map is a **callable that creates
one** — `digit_classifier.cpp`:

```cpp
using Factory = std::function<std::unique_ptr<DigitClassificationInterface>()>;

const std::map<std::string, Factory>& registry() {
    static const std::map<std::string, Factory> table = {
        {"cnn", [] { return std::make_unique<CnnModel>(); }},
        {"rf", [] { return std::make_unique<RandomForestModel>(); }},
        {"rand", [] { return std::make_unique<RandomModel>(); }},
    };
    return table;
}
```

`[] { return std::make_unique<CnnModel>(); }` is a **lambda** — an
anonymous, inline function — the C++ equivalent of a tiny
`lambda: CnnModel()` in Python, except C++'s lambdas can be full
multi-line function bodies, not just a single expression.
`std::function<std::unique_ptr<DigitClassificationInterface>()>` is the
*type* of "a callable that takes no arguments and returns a
`unique_ptr<DigitClassificationInterface>`" — `std::map<string, Factory>`
is then exactly `_ALGORITHMS` from the Python version, just with
"a class" replaced by "a function that builds one," because that's the
nearest thing C++ has to a class being storable as a value.

`registry()` returns the table by `const&` from inside a function with a
`static const` local — the table is built exactly once, the first time
`registry()` is called, and reused after that (same lifetime as a
module-level dict in Python, just constructed lazily instead of at
import time).

## 5. Recursive ownership: `RandomForestModel`'s decision trees

The Python version's `RandomForestModel` leans on scikit-learn for the
actual tree structure. Here, `decision_tree.hpp` builds one from scratch —
a genuine (if depth-limited, randomly-searched) CART-style tree, useful
as a second, more elaborate demonstration of `unique_ptr`:

```cpp
struct TreeNode {
    bool is_leaf = false;
    int predicted_class = 0;
    int feature_index = -1;
    double threshold = 0.0;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};
```

A node **owns** its two children. Destroy the root node, and its
`unique_ptr<TreeNode> left`'s destructor runs, which destroys *that*
node's children, and so on, all the way down — one line of ownership
(`std::unique_ptr`) gives you correct, automatic, recursive cleanup of an
entire tree with no explicit cleanup code anywhere. This is the same RAII
idea as `std::vector` freeing its buffer, just applied to a
pointer-based, recursive structure instead of a flat one.

Traversal (`DecisionTree::predict`), by contrast, uses a **raw, non-owning
pointer**:

```cpp
const TreeNode* node = root_.get();
while (!node->is_leaf) {
    node = (x[node->feature_index] <= node->threshold) ? node->left.get() : node->right.get();
}
```

`.get()` reaches into a `unique_ptr` and hands out the raw address it
holds, *without* transferring ownership. This is the key distinction to
walk away with: a **raw pointer is fine and normal in modern C++ when
you're just looking, not responsible for freeing anything**; you reach
for a smart pointer specifically for the variable that *owns* the thing
and must eventually destroy it. `node` here is never responsible for
deleting anything it points to — `root_` (and the `left`/`right`
`unique_ptr`s hanging off each node) are.

`DecisionTree::fit` builds the tree by picking a handful of random
`(feature, threshold)` candidates per node and keeping whichever one cuts
Gini impurity the most — a real, if simplified, decision-tree training
loop (a full implementation would exhaustively scan every possible split
point on every feature; scanning a random sample of candidates instead is
the deliberate size/complexity trade-off made here). `RandomForestModel`
builds 10 of these on the same tiny random "warm start" dataset the
Python version uses (a couple of random points per digit, purely so
`predict()` has *something* to traverse — see the comment in
`random_forest_model.cpp`), and predicts by majority vote across all 10
trees.

## 6. The CNN: a real (untrained) forward pass, no ML library

Same spirit as the Python version's CNN: `train()` is out of scope (see
section 7), so the weights are random and never updated — but `predict()`
still has to run a real forward pass through a real, if tiny, 2-conv-layer
network (`cnn_model.cpp`, mirroring `cnn_model.py`'s PyTorch architecture
exactly: `Conv(1→8) → ReLU → MaxPool → Conv(8→16) → ReLU → MaxPool →
Linear(784→10) → argmax`). `ConvLayer` and `LinearLayer`
(`layers.hpp`/`.cpp`) are small classes that own their own randomly
initialized weights and know how to apply themselves to an input — the
same idea as PyTorch's `nn.Conv2d`/`nn.Linear`, just hand-written, with
plain nested loops instead of a tensor library. `Tensor3D` (a
`vector<vector<vector<float>>>`, `[channel][row][col]`) intentionally
uses nested vectors rather than the flat single-buffer trick from
`cpp/regression/`'s `Matrix` — these tensors are tiny (at most `8×28×28`),
so the extra clarity of nested indexing was worth more here than the
performance `Matrix` needed for a `90000×55` design matrix.

## 7. `throw std::logic_error(...)` — C++'s answer to `NotImplementedError`

Python has a dedicated `NotImplementedError` exception type for exactly
this situation. C++'s standard library doesn't have a same-named
equivalent, but it does have a small family of exception types under
`<stdexcept>` for different *kinds* of problems, and the closest match
here is `std::logic_error` — its whole category is "a violation of a
precondition or contract that a careful caller could have checked for in
advance," which is precisely what "you called `train()`, but this
implementation deliberately doesn't support that" is. Every model's
`train()` override does the same thing:

```cpp
void CnnModel::train(const std::vector<std::vector<double>>&, const std::vector<int>&) {
    throw std::logic_error("Training is out of scope for this exercise.");
}
```

(Notice the parameter names are dropped — `const std::vector<...>&` with
no name after it. Legal in C++ when a parameter must be present to match
the overridden signature but the function body never uses it; it silences
the "unused parameter" warning some compilers would otherwise raise.)

An uncaught `std::logic_error` (or any uncaught exception) terminates the
program with an error message — same end result as an uncaught Python
exception printing a traceback and exiting, just a shorter message by
default. `test_mnist.cpp` shows the *caught* case: `catch (const
std::logic_error&)`, the direct equivalent of Python's `except
NotImplementedError:`.

## 8. Building and running it

```bash
cd cpp/mnist
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/demo          # runs all three algorithms on one random image
./build/test_mnist    # unit tests
```

## 9. Side-by-side with the Python version

| | Python (`mnist/`) | C++ (`cpp/mnist/`) |
|---|---|---|
| Interface | `ABC` + `@abstractmethod` | abstract class, pure virtual (`= 0`) functions |
| "Is a" relationship | `class CnnModel(DigitClassificationInterface):` | `class CnnModel : public DigitClassificationInterface` |
| Dynamic dispatch | automatic, always on | opt-in via `virtual` (vtable lookup at runtime) |
| Owning a polymorphic object | `self._model = model_cls()` (GC frees it) | `std::unique_ptr<DigitClassificationInterface>` (deterministic, automatic `delete`) |
| Registry of algorithms | `dict` of classes | `std::map` of factory lambdas (`std::function`) |
| "Not implemented" | `raise NotImplementedError` | `throw std::logic_error(...)` |
| Tree structure | `sklearn.ensemble.RandomForestClassifier` | hand-rolled `TreeNode` tree, `unique_ptr`-owned |
| CNN | `torch.nn.Module` (random, untrained weights) | hand-rolled `ConvLayer`/`LinearLayer` (random, untrained weights) |

The Strategy pattern (`DigitClassifier` *holds* a model, doesn't *inherit*
from the interface) and the Adapter logic (`adapt()` reshaping/cropping
the canonical image per algorithm) are structurally identical to the
Python version — same reasoning applies, see the Python explanation for
*why* composition beats inheritance here. What changed going from Python
to C++ is entirely the machinery required to make "one interface, many
implementations, chosen at runtime" *possible* in a language where types
are fixed at compile time — `virtual`, `unique_ptr`, and `std::function`
are that machinery.
