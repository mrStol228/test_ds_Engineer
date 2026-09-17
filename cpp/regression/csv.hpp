#pragma once

#include <string>
#include <vector>

struct CsvTable {
    std::vector<std::string> header;
    std::vector<std::vector<double>> rows;
};

CsvTable read_csv(const std::string& path);
