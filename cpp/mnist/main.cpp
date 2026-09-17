#include <iostream>
#include <random>

#include "digit_classifier.hpp"

int main() {
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<double> image(DigitClassifier::kCanonicalLength);
    for (double& v : image) {
        v = dist(rng);
    }

    for (const std::string& algorithm : {"cnn", "rf", "rand"}) {
        DigitClassifier classifier(algorithm);
        int prediction = classifier.predict(image);
        std::cout << algorithm << ": predicted digit = " << prediction << std::endl;
    }

    return 0;
}
