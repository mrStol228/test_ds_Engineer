#include "decision_tree.hpp"

#include <map>
#include <numeric>

double DecisionTree::gini(const std::vector<size_t>& indices, const std::vector<int>& y) {
    std::map<int, int> counts;
    for (size_t i : indices) {
        counts[y[i]]++;
    }
    double n = static_cast<double>(indices.size());
    double impurity = 1.0;
    for (const auto& [cls, count] : counts) {
        double p = count / n;
        impurity -= p * p;
    }
    return impurity;
}

int DecisionTree::majority_class(const std::vector<size_t>& indices, const std::vector<int>& y) {
    std::map<int, int> counts;
    for (size_t i : indices) {
        counts[y[i]]++;
    }
    int best_class = 0;
    int best_count = -1;
    for (const auto& [cls, count] : counts) {
        if (count > best_count) {
            best_count = count;
            best_class = cls;
        }
    }
    return best_class;
}

std::unique_ptr<TreeNode> DecisionTree::build(const std::vector<size_t>& indices,
                                               const std::vector<std::vector<double>>& X,
                                               const std::vector<int>& y, std::mt19937& rng, int depth,
                                               int max_depth) {
    auto node = std::make_unique<TreeNode>();

    double parent_impurity = gini(indices, y);
    if (depth >= max_depth || indices.size() <= 1 || parent_impurity < 1e-9) {
        node->is_leaf = true;
        node->predicted_class = majority_class(indices, y);
        return node;
    }

    size_t n_features = X[0].size();
    std::uniform_int_distribution<size_t> feature_dist(0, n_features - 1);
    std::uniform_int_distribution<size_t> row_dist(0, indices.size() - 1);

    double best_gain = 0.0;
    int best_feature = -1;
    double best_threshold = 0.0;
    std::vector<size_t> best_left, best_right;

    // Try a handful of random (feature, threshold) candidates rather than
    // exhaustively searching every feature and every possible split point --
    // a deliberate simplification, see README.md.
    const int kCandidates = 8;
    for (int attempt = 0; attempt < kCandidates; ++attempt) {
        size_t feature = feature_dist(rng);
        size_t pivot_row = indices[row_dist(rng)];
        double threshold = X[pivot_row][feature];

        std::vector<size_t> left, right;
        for (size_t i : indices) {
            (X[i][feature] <= threshold ? left : right).push_back(i);
        }
        if (left.empty() || right.empty()) {
            continue;
        }

        double weighted = (left.size() * gini(left, y) + right.size() * gini(right, y)) /
                           static_cast<double>(indices.size());
        double gain = parent_impurity - weighted;
        if (gain > best_gain) {
            best_gain = gain;
            best_feature = static_cast<int>(feature);
            best_threshold = threshold;
            best_left = std::move(left);
            best_right = std::move(right);
        }
    }

    if (best_feature == -1) {
        node->is_leaf = true;
        node->predicted_class = majority_class(indices, y);
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = build(best_left, X, y, rng, depth + 1, max_depth);
    node->right = build(best_right, X, y, rng, depth + 1, max_depth);
    return node;
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, std::mt19937& rng,
                        int max_depth) {
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    root_ = build(indices, X, y, rng, 0, max_depth);
}

int DecisionTree::predict(const std::vector<double>& x) const {
    const TreeNode* node = root_.get();  // observing, non-owning pointer -- see README.md
    while (!node->is_leaf) {
        node = (x[node->feature_index] <= node->threshold) ? node->left.get() : node->right.get();
    }
    return node->predicted_class;
}
