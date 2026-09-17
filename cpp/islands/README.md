# Counting Islands in C++ — a from-scratch intro

This is the same task as `islands/` (the Python version), rewritten in C++.
Since you haven't touched C++ before, this file walks through *every* new
concept it uses, comparing each one directly to the Python you already know.
Read this one first — `regression/README.md` and `mnist/README.md` build on
it and only explain what's new there.

## 1. C++ is not interpreted — it's compiled

In Python, `python3 islands.py` reads your source file and runs it line by
line, on the spot. There is no such step in C++. Before anything can run,
a separate program (the **compiler**) translates your `.cpp` files into
machine code, and another step (the **linker**) glues the pieces into one
executable file. Only *that* file runs.

```
islands.cpp  --[compiler]-->  islands.o   (object file: machine code, but not runnable yet)
main.cpp     --[compiler]-->  main.o
islands.o + main.o  --[linker]-->  islands   (the actual runnable program)
```

This is why there's a `build/` step before you can run anything — see
section 8. It's also why C++ mistakes fall into two very different buckets:

- **Compile errors** — the compiler refuses to translate your code at all
  (e.g. a typo in a type name, a missing semicolon). Nothing runs.
- **Runtime errors** — the code compiles fine but misbehaves when executed
  (e.g. reading past the end of an array). C++ often won't even tell you —
  more on this in section 6.

Python only really has the second kind (`SyntaxError` at import time is the
closest thing to a compile error, but it's still checked line-by-line at
runtime by the interpreter, not ahead of time by a separate program).

## 2. Every variable has a fixed, explicit type

Python:
```python
grid = [[0, 1], [1, 0]]   # grid can later be reassigned to a string, an int, anything
islands = 0
```

C++ (`islands.cpp`):
```cpp
Grid grid;        // Grid is declared elsewhere to MEAN std::vector<std::vector<int>>
int islands = 0;  // islands is an int, forever, for the rest of its scope
```

Once `islands` is declared as `int`, the compiler will reject `islands =
"two";` at compile time — not at runtime like Python's duck typing would
allow (Python wouldn't even complain about that reassignment; it would just
silently change what `islands` refers to). This is the trade C++ makes:
more upfront ceremony, in exchange for an entire category of bugs being
caught before the program ever runs.

`using Grid = std::vector<std::vector<int>>;` in `islands.hpp` is a **type
alias** — it doesn't create a new type, it just gives a long, nested type a
short name, purely for readability. Comparable to `Grid = list[list[int]]`
if you've ever used Python's `typing` module — except in C++ the alias is
enforced by the compiler, not just a hint for readers/type-checkers.

## 3. `std::vector` is Python's `list`, mostly

`std::vector<int>` is a growable array of `int`s — the closest C++ analog
to a Python `list`, but with one big difference: **a `std::vector<int>` can
only ever hold `int`s.** Python's `list` can mix a string, an int, and a
dict in the same list; `std::vector` cannot mix types — that's what "every
variable has a fixed type" (section 2) means applied to containers.

```cpp
std::vector<int> row = {0, 1, 0};   // like row = [0, 1, 0] in Python
row[0]              // indexing works the same way: row[0] -> 0
row.size()           // len(row) in Python
row.push_back(1)     // row.append(1) in Python
```

`Grid` is a `vector` of `vector<int>` — a 2-D grid, same idea as a Python
list of lists, just statically typed two levels deep.

## 4. Pass-by-value vs. pass-by-reference — the biggest Python trap

This is the single most important difference to internalize, because it's
invisible if you don't know to look for it.

In Python, when you pass a list into a function, the function gets a
reference to the *same* list — mutating it inside the function mutates the
caller's list too. That's exactly what the Python solution relies on:
`count_islands(grid)` sets cells to `0` in place, and the caller's `grid`
ends up fully zeroed out by the time the function returns.

In C++, **passing by value is the default**. `count_islands(Grid grid)`
means the function receives its own independent **copy** of the grid.
Mutating `grid` inside the function does nothing to whatever the caller
passed in — the caller's grid is untouched. That's why the tests in
`test_islands.cpp` can call `count_islands({{1, 0}, {0, 1}})` — a fresh
literal each time — without worrying about a previous call having zeroed
it out.

If C++ wanted Python's aliasing behavior, the signature would instead be
`int count_islands(Grid& grid)` — the `&` makes it a **reference**: no
copy, the function operates directly on the caller's data, and any
mutation is visible outside. `read_grid(std::istream& in)` uses exactly
this: `in` is a reference to the actual input stream (`std::cin` or a
file), not a copy of it — copying an open file handle wouldn't even make
sense.

Rule of thumb while reading any C++ signature: `Type x` copies, `Type& x`
aliases (and can mutate the caller's variable), `const Type& x` aliases
but promises not to mutate it (a very common combination — see it appear
throughout `regression/`).

## 5. `std::queue` is `collections.deque`, restricted to a queue

The Python solution uses `collections.deque` for BFS because it gives O(1)
appends/pops from both ends. C++'s `std::queue<T>` is a narrower tool: it
only exposes `push` (add to the back), `front` (peek at the front) and
`pop` (remove the front) — enough for a queue, nothing more. It's built on
top of another container internally (by default `std::deque`), which is
why it behaves efficiently.

```cpp
std::queue<std::pair<int, int>> queue;
queue.push({r, c});          // append((r, c)) in Python
auto [cr, cc] = queue.front();  // peek without removing
queue.pop();                 // popleft() -- but note: pop() here does NOT return the value!
```

That last point trips people up: unlike Python's `deque.popleft()`, C++'s
`queue.pop()` returns *nothing* — you have to `front()` first to read the
value, then `pop()` separately to remove it. Two steps where Python needed
one.

## 6. `std::pair` and structured bindings

`std::pair<int, int>` bundles two values together — like a 2-tuple in
Python. `queue.front()` returns one; to pull it apart into two named
variables, C++17 added **structured bindings**:

```cpp
auto [cr, cc] = queue.front();
```

is the direct C++ equivalent of Python's `cr, cc = queue[0]`. `auto` here
means "figure out the type yourself" (it's `std::pair<int, int>`, but
writing that out twice would be noise) — `auto` is one of the few places
C++ lets you skip stating the type explicitly, and it's purely a
convenience for the *writer*; the compiler still pins down a single,
fixed, concrete type at compile time. It is not the same thing as Python's
dynamic typing (section 2) — `auto x = 5;` makes `x` an `int` forever,
it just saves you from typing `int` yourself.

## 7. No automatic bounds checking

Python: `row[5]` on a 3-element list raises `IndexError` — always, no
exceptions (pun intended).

C++: `row[5]` on a 3-element `vector` is **undefined behavior** by
default — it might crash, might silently read garbage memory, might
"work" and corrupt something else entirely. The compiler does not insert
a bounds check for `operator[]`, because that check costs time and C++'s
whole design philosophy is "don't pay for what you don't use."

This is exactly why `count_islands` checks
`0 <= nr && nr < rows && 0 <= nc && nc < cols` **before** touching
`grid[nr][nc]` — in the Python version this guard exists too, but in
Python it's a matter of taste; in C++ it's load-bearing. Skipping it isn't
a clean `IndexError` you can catch — it's a landmine.

(If you want checked access, `vector` does have `.at(5)`, which *does*
throw `std::out_of_range` — but it's not the default `[]` operator, and
it's slower, so idiomatic C++ tends to prove correctness with guards like
the one above rather than lean on exceptions for routine bounds checks.)

## 8. Building and running it

```bash
cd cpp/islands
cmake -S . -B build      # "configure": figure out compiler, generate build files
cmake --build build      # "build": actually compile + link

./build/islands input.txt
printf "3 4\n0 0 0 1\n0 0 1 1\n0 1 0 1\n" | ./build/islands
./build/test_islands
```

`CMakeLists.txt` is not C++ — it's a separate, much simpler language that
describes *how* to build the project (which files, which flags, what to
name the output) so you don't have to type raw `g++` compiler invocations
by hand every time. It's the closest thing C++ has to `requirements.txt` +
`setup.py` combined, except it's about compiling, not installing packages.
`build/` is generated output — never edit anything inside it, and it's
git-ignored for the same reason `.venv/` is in the Python projects: it's
reconstructible from source, so it doesn't belong in version control.

## 9. Side-by-side with the Python version

| | Python (`islands/islands.py`) | C++ (`cpp/islands/`) |
|---|---|---|
| Run it | `python3 islands.py` | compile first, then run the binary |
| Types | dynamic, checked at runtime | static, checked at compile time |
| List | `list`, mixed types allowed | `std::vector<T>`, one type only |
| Passing a list to a function | always aliases (shares memory) | copies by default; `&` to alias |
| Queue | `collections.deque` | `std::queue` |
| Tuple unpack | `a, b = pair` | `auto [a, b] = pair;` (C++17) |
| Out-of-bounds access | `IndexError` | undefined behavior — your job to prevent it |
| "Run the file" | one step | compile, then link, then run |

The algorithm itself — BFS flood fill, 4-connectivity, `O(rows*cols)` — is
identical line-for-line in spirit to `islands/islands.py`; only the
language's rules around it changed.
