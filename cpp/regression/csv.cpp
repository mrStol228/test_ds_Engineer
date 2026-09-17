#include "csv.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

}  // namespace

CsvTable read_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }

    CsvTable table;

    std::string header_line;
    std::getline(file, header_line);
    table.header = split_line(header_line);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_line(line);
        std::vector<double> row;
        row.reserve(fields.size());
        for (const std::string& field : fields) {
            row.push_back(std::stod(field));
        }
        table.rows.push_back(std::move(row));
    }

    return table;
}
