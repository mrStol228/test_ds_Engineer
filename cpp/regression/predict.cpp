#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"
#include "csv.hpp"
#include "linear_regression.hpp"

int main(int argc, char* argv[]) {
    std::string model_path = argc > 1 ? argv[1] : "models/weights.txt";
    std::string input_path = argc > 2 ? argv[2] : "../../regression/data/hidden_test.csv";
    std::string output_path = argc > 3 ? argv[3] : "predictions/hidden_test_predictions.csv";

    std::cout << "Loading weights from " << model_path << " ..." << std::endl;
    std::vector<double> weights = load_weights(model_path);
    if (weights.size() != DESIGN_COLUMN_COUNT) {
        std::cerr << "Expected " << DESIGN_COLUMN_COUNT << " weights, got " << weights.size() << std::endl;
        return 1;
    }

    std::cout << "Reading " << input_path << " ..." << std::endl;
    CsvTable table = read_csv(input_path);
    size_t n = table.rows.size();
    std::cout << "Loaded " << n << " rows." << std::endl;

    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Could not open " << output_path << " for writing" << std::endl;
        return 1;
    }
    out.precision(17);
    out << "id,target\n";

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> raw(table.rows[i].begin(), table.rows[i].begin() + RAW_FEATURE_COUNT);
        std::vector<double> engineered = engineer_features(raw);
        double prediction = predict_row(weights, engineered);
        out << i << ',' << prediction << '\n';
    }

    std::cout << "Saved " << n << " predictions to " << output_path << std::endl;
    return 0;
}
