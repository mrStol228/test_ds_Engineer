#include <iostream>
#include <random>
#include <set>

#include "digit_classifier.hpp"
#include "random_model.hpp"

namespace {

int failures = 0;

void check(const std::string& name, bool ok) {
    if (ok) {
        std::cout << "PASS  " << name << std::endl;
    } else {
        std::cout << "FAIL  " << name << std::endl;
        ++failures;
    }
}

std::vector<double> random_image() {
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> image(DigitClassifier::kCanonicalLength);
    for (double& v : image) {
        v = dist(rng);
    }
    return image;
}

}  // namespace

int main() {
    std::vector<double> image = random_image();

    for (const std::string& algorithm : {"cnn", "rf", "rand"}) {
        DigitClassifier classifier(algorithm);
        int prediction = classifier.predict(image);
        check("predict_returns_valid_digit[" + algorithm + "]", prediction >= 0 && prediction <= 9);
    }

    {
        bool threw = false;
        try {
            DigitClassifier bad("svm");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check("unknown_algorithm_throws", threw);
    }

    for (const std::string& algorithm : {"cnn", "rf", "rand"}) {
        DigitClassifier classifier(algorithm);
        bool rejected = false;
        try {
            classifier.predict(std::vector<double>(100, 0.0));  // wrong length, not 784
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check("predict_rejects_wrong_shape[" + algorithm + "]", rejected);
    }

    for (const std::string& algorithm : {"cnn", "rf", "rand"}) {
        DigitClassifier classifier(algorithm);
        bool threw = false;
        try {
            classifier.train({}, {});
        } catch (const std::logic_error&) {
            threw = true;
        }
        check("train_not_implemented[" + algorithm + "]", threw);
    }

    {
        RandomModel random_model(42);
        std::set<int> seen;
        std::vector<double> crop(RandomModel::kInputLength, 0.0);
        for (int i = 0; i < 2000; ++i) {
            seen.insert(random_model.predict(crop));
        }
        check("random_model_covers_all_ten_digits", seen.size() == 10);
    }

    if (failures == 0) {
        std::cout << "\nAll tests passed." << std::endl;
        return 0;
    }
    std::cout << "\n" << failures << " test(s) failed." << std::endl;
    return 1;
}
