#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

// This is a revisit of the class project that I could never complete
// I will make yet another attempt at making it work.
// I wrote this in python as a tutorial, I have to remember I am using
// pointers now. Need to keep track of those.

struct Node
{
    bool is_leaf;
    int feature;
    double threshold;
    double value;
    Node *left;
    Node *right;

    Node(bool is_leaf = false, int feature = -1, double threshold = 0, double value = 0)
        : is_leaf(is_leaf), feature(feature), threshold(threshold), value(value), left(nullptr), right(nullptr) {}
};

// Calculate Entropy
// This function returns a double.
double entropy(const std::vector<int> &labels)
{
    std::map<int, int> label_count;
    for (int label : labels)
    {
        label_count[label]++;
    }

    double entropy = 0.0;
    for (const auto &[label, count] : label_count)
    {
        // P is probability, the calculation of entropy is
        // sum(probability * log2(probability))
        double p = static_cast<double>(count) / labels.size();
        entropy -= p * std::log2(p);
    }
    return entropy;
}

// Split the datasets into left and right
// Return type is a vector or two vectors

std::pair<std::vector<int>, std::vector<int>> split(const std::vector<double> &feature_values, double threshold)
{
    std::vector<int> left_idxs, right_idxs;
    for (size_t i = 0; i < feature_values.size(); ++i)
    {
        if (feature_values[i] <= threshold)
        {
            left_idxs.push_back(i);
        }
        else
        {
            right_idxs.push_back(i);
        }
    }
    return {left_idxs, right_idxs};
}

// Build the tree

Node *build_tree(const std::vector<std::vector<double>> &X, const std::vector<int> &y, int depth = 0)
{
    if (X.empty() || y.empty())
    {
        return new Node(true, -1, 0, y[0]);
    }

    double best_gain = -std::numeric_limits<double>::infinity();
    int best_feature = -1;
    double best_threshold = 0;

    for (size_t i = 0; i < X[0].size(); ++i)
    {
        std::vector<double> feature_values;
        for (const auto &row : X)
        {
            feature_values.push_back(row[i]);
        }

        std::vector<double> unique_thresholds(feature_values.begin(), feature_values.end());
        std::sort(unique_thresholds.begin(), unique_thresholds.end());
        unique_thresholds.erase(std::unique(unique_thresholds.begin(), unique_thresholds.end()), unique_thresholds.end());

        for (double threshold : unique_thresholds)
        {
            auto [left_idxs, right_idxs] = split(feature_values, threshold);
            if (left_idxs.empty() || right_idxs.empty())
            {
                continue;
            }

            std::vector<int> left_labels, right_labels;
            for (int idx : left_idxs)
                left_labels.push_back(y[idx]);
            for (int idx : right_idxs)
                right_labels.push_back(y[idx]);

            double left_entropy = entropy(left_labels);
            double right_entropy = entropy(right_labels);

            double info_gain = entropy(y) - (left_labels.size() * left_entropy + right_labels.size() * right_entropy) / y.size();

            if (info_gain > best_gain)
            {
                best_gain = info_gain;
                best_feature = i;
                best_threshold = threshold;
            }
        }
    }

    if (best_gain == -std::numeric_limits<double>::infinity())
    {
        return new Node(true, -1, 0, y[0]);
    }

    std::vector<double> best_feature_values;
    for (const auto &row : X)
        best_feature_values.push_back(row[best_feature]);
    auto [left_idxs, right_idxs] = split(best_feature_values, best_threshold);

    std::vector<std::vector<double>> X_left, X_right;
    std::vector<int> y_left, y_right;

    for (int idx : left_idxs)
    {
        X_left.push_back(X[idx]);
        y_left.push_back(y[idx]);
    }
    for (int idx : right_idxs)
    {
        X_right.push_back(X[idx]);
        y_right.push_back(y[idx]);
    }

    Node *left_child = build_tree(X_left, y_left, depth + 1);
    Node *right_child = build_tree(X_right, y_right, depth + 1);

    Node *node = new Node(false, best_feature, best_threshold, 0);
    node->left = left_child;
    node->right = right_child;

    return node;
}

int predict(const std::vector<double> &instance, Node *node)
{
    if (node->is_leaf)
    {
        return static_cast<int>(node->value);
    }

    if (instance[node->feature] <= node->threshold)
    {
        return predict(instance, node->left);
    }
    else
    {
        return predict(instance, node->right);
    }
}

int main()
{
    std::vector<std::vector<double>> X = {
        {2.0, 3.0}, {1.0, 5.0}, {3.0, 2.0}, {2.0, 4.0}, {5.0, 2.0}, {6.0, 5.0}, {7.0, 3.0}, {8.0, 4.0}};
    std::vector<int> y = {0, 1, 0, 1, 0, 1, 0, 1};

    Node *root = build_tree(X, y);
    std::vector<double> new_instance = {6.0, 4.0};
    int prediction = predict(new_instance, root);

    std::cout << "Prediction: " << prediction << std::endl;

    delete root;
    return 0;
}