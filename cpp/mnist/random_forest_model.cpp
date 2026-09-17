#include "random_forest_model.hpp"

#include <map>
#include <stdexcept>

RandomForestModel::RandomForestModel() : rng_(42) {
    warm_start();
}

void RandomForestModel::warm_start() {
    // `train()` is deliberately not implemented (see below). scikit-learn's
    // RandomForestClassifier needs at least one fit() call before predict()
    // works at all -- the Python version handles that by fitting on a
    // handful of random points covering all 10 digits. Same idea here: this
    // is not training for quality, just enough structure to make predict()
    // callable.
    std::uniform_real_distribution<double> value_dist(0.0, 1.0);

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int digit = 0; digit < 10; ++digit) {
            std::vector<double> row(kInputLength);
            for (double& v : row) {
                v = value_dist(rng_);
            }
            X.push_back(std::move(row));
            y.push_back(digit);
        }
    }

    const int kTreeCount = 10;
    trees_.resize(kTreeCount);
    for (DecisionTree& tree : trees_) {
        tree.fit(X, y, rng_, /*max_depth=*/3);
    }
}

void RandomForestModel::train(const std::vector<std::vector<double>>&, const std::vector<int>&) {
    throw std::logic_error("Training is out of scope for this exercise.");
}

int RandomForestModel::predict(const std::vector<double>& image) {
    if (image.size() != kInputLength) {
        throw std::invalid_argument("RandomForestModel::predict expects 784 values");
    }

    std::map<int, int> votes;
    for (const DecisionTree& tree : trees_) {
        votes[tree.predict(image)]++;
    }

    int best_class = 0;
    int best_votes = -1;
    for (const auto& [cls, count] : votes) {
        if (count > best_votes) {
            best_votes = count;
            best_class = cls;
        }
    }
    return best_class;
}
