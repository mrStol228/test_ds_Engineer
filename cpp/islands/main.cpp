#include <fstream>
#include <iostream>

#include "islands.hpp"

int main(int argc, char* argv[]) {
    Grid grid;

    if (argc > 1) {
        std::ifstream file(argv[1]);
        if (!file) {
            std::cerr << "Could not open file: " << argv[1] << std::endl;
            return 1;
        }
        grid = read_grid(file);
    } else {
        grid = read_grid(std::cin);
    }

    std::cout << count_islands(grid) << std::endl;
    return 0;
}
