import sys
from collections import deque
from typing import List, TextIO

Grid = List[List[int]]


def count_islands(grid: Grid) -> int:
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1:
                continue

            islands += 1
            grid[r][c] = 0
            queue = deque([(r, c)])

            while queue:
                cr, cc = queue.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 0
                        queue.append((nr, nc))

    return islands


def read_grid(stream: TextIO) -> Grid:
    m, n = map(int, stream.readline().split())
    return [list(map(int, stream.readline().split())) for _ in range(m)]


def main() -> None:
    source = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    try:
        grid = read_grid(source)
    finally:
        if source is not sys.stdin:
            source.close()

    print(count_islands(grid))


if __name__ == "__main__":
    main()
