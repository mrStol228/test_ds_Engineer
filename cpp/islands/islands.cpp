#include "islands.hpp"

#include <queue>
#include <utility>

int count_islands(Grid grid) {
    if (grid.empty() || grid[0].empty()) {
        return 0;
    }

    int rows = static_cast<int>(grid.size());
    int cols = static_cast<int>(grid[0].size());
    int islands = 0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] != 1) {
                continue;
            }

            ++islands;
            grid[r][c] = 0;

            std::queue<std::pair<int, int>> queue;
            queue.push({r, c});

            while (!queue.empty()) {
                auto [cr, cc] = queue.front();
                queue.pop();

                static const int dr[] = {-1, 1, 0, 0};
                static const int dc[] = {0, 0, -1, 1};

                for (int d = 0; d < 4; ++d) {
                    int nr = cr + dr[d];
                    int nc = cc + dc[d];
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                        grid[nr][nc] = 0;
                        queue.push({nr, nc});
                    }
                }
            }
        }
    }

    return islands;
}

Grid read_grid(std::istream& in) {
    int rows = 0;
    int cols = 0;
    in >> rows >> cols;

    Grid grid(rows, std::vector<int>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            in >> grid[r][c];
        }
    }
    return grid;
}
