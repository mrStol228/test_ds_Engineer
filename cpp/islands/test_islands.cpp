#include <iostream>

#include "islands.hpp"

namespace {

int failures = 0;

void check(const std::string& name, int actual, int expected) {
    if (actual == expected) {
        std::cout << "PASS  " << name << std::endl;
    } else {
        std::cout << "FAIL  " << name << "  (got " << actual << ", expected " << expected << ")" << std::endl;
        ++failures;
    }
}

}  // namespace

int main() {
    check("task_case_1", count_islands({{0, 1, 0}, {0, 0, 0}, {0, 1, 1}}), 2);
    check("task_case_2", count_islands({{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}}), 3);
    check("task_case_3", count_islands({{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 0, 1}}), 2);

    check("empty_grid", count_islands({}), 0);
    check("all_ocean", count_islands({{0, 0}, {0, 0}}), 0);
    check("all_land", count_islands({{1, 1}, {1, 1}}), 1);
    check("single_cell", count_islands({{1}}), 1);
    check("diagonal_not_connected", count_islands({{1, 0}, {0, 1}}), 2);

    if (failures == 0) {
        std::cout << "\nAll tests passed." << std::endl;
        return 0;
    }
    std::cout << "\n" << failures << " test(s) failed." << std::endl;
    return 1;
}
