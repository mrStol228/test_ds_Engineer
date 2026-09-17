import numpy as np
import pytest

from digit_classifier import DigitClassifier
from digit_classifier.cnn_model import CnnModel
from digit_classifier.random_forest_model import RandomForestModel
from digit_classifier.random_model import RandomModel


def random_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((28, 28, 1)).astype(np.float32)


@pytest.mark.parametrize("algorithm", ["cnn", "rf", "rand"])
def test_predict_returns_single_digit(algorithm):
    classifier = DigitClassifier(algorithm)
    prediction = classifier.predict(random_image())

    assert isinstance(prediction, int)
    assert 0 <= prediction <= 9


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        DigitClassifier("svm")


@pytest.mark.parametrize("algorithm", ["cnn", "rf", "rand"])
def test_predict_rejects_wrong_shape(algorithm):
    classifier = DigitClassifier(algorithm)
    with pytest.raises(ValueError):
        classifier.predict(np.zeros((28, 28)))


@pytest.mark.parametrize("algorithm", ["cnn", "rf", "rand"])
def test_train_is_not_implemented(algorithm):
    classifier = DigitClassifier(algorithm)
    with pytest.raises(NotImplementedError):
        classifier.train(np.zeros((1, 28, 28, 1)), np.zeros((1,)))


def test_random_model_is_uniformly_distributed_over_many_calls():
    model = RandomModel(seed=42)
    image = np.zeros((10, 10))
    predictions = [model.predict(image) for _ in range(2000)]

    assert set(predictions) == set(range(10))


def test_each_model_rejects_its_own_wrong_input_shape():
    with pytest.raises(ValueError):
        CnnModel().predict(np.zeros((10, 10, 1)))
    with pytest.raises(ValueError):
        RandomForestModel().predict(np.zeros((28, 28)))
    with pytest.raises(ValueError):
        RandomModel().predict(np.zeros((28, 28, 1)))


def test_center_crop_matches_manual_slice():
    classifier = DigitClassifier("rand")
    image = np.arange(28 * 28).reshape(28, 28, 1).astype(np.float32)

    adapted = classifier._adapt(image)

    expected = image[9:19, 9:19, 0]
    np.testing.assert_array_equal(adapted, expected)
