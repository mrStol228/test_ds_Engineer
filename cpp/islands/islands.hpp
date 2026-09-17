#pragma once

#include <iostream>
#include <vector>

using Grid = std::vector<std::vector<int>>;

int count_islands(Grid grid);
Grid read_grid(std::istream& in);
