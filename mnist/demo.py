import numpy as np

from digit_classifier import DigitClassifier


def main() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((28, 28, 1)).astype(np.float32)

    for algorithm in ("cnn", "rf", "rand"):
        classifier = DigitClassifier(algorithm)
        prediction = classifier.predict(image)
        print(f"{algorithm:>5}: predicted digit = {prediction}")


if __name__ == "__main__":
    main()
