#pragma once

#include <memory>
#include <random>
#include <vector>

// A tree node OWNS its children -- when a node is destroyed, its unique_ptr
// members destroy their subtrees automatically, recursively. See README.md.
struct TreeNode {
    bool is_leaf = false;
    int predicted_class = 0;
    int feature_index = -1;
    double threshold = 0.0;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};

class DecisionTree {
public:
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, std::mt19937& rng, int max_depth = 3);
    int predict(const std::vector<double>& x) const;

private:
    std::unique_ptr<TreeNode> root_;

    static std::unique_ptr<TreeNode> build(const std::vector<size_t>& indices,
                                            const std::vector<std::vector<double>>& X,
                                            const std::vector<int>& y, std::mt19937& rng, int depth,
                                            int max_depth);
    static double gini(const std::vector<size_t>& indices, const std::vector<int>& y);
    static int majority_class(const std::vector<size_t>& indices, const std::vector<int>& y);
};
