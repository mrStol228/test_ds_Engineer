# Counting Islands

BFS flood fill over a binary grid, `O(M*N)`. Counts 4-connected groups of `1`s
(diagonal neighbors don't count — matters for the second test case).

Run it:

    python3 islands.py input.txt
    # or through stdin
    printf "3 4\n0 0 0 1\n0 0 1 1\n0 1 0 1\n" | python3 islands.py

Input is `M N` on the first line, then `M` rows of the grid.

Tests: `python3 -m pytest test_islands.py -v` — the three cases from the
task description plus some edge cases (empty grid, all ocean, all land,
single cell, diagonal cells that shouldn't merge into one island).
