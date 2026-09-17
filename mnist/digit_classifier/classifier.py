from typing import Dict, Type

import numpy as np

from .cnn_model import CnnModel
from .interface import DigitClassificationInterface
from .random_forest_model import RandomForestModel
from .random_model import RandomModel

_ALGORITHMS: Dict[str, Type[DigitClassificationInterface]] = {
    "cnn": CnnModel,
    "rf": RandomForestModel,
    "rand": RandomModel,
}

CANONICAL_INPUT_SHAPE = (28, 28, 1)


class DigitClassifier:
    def __init__(self, algorithm: str) -> None:
        try:
            model_cls = _ALGORITHMS[algorithm]
        except KeyError as exc:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. Available: {sorted(_ALGORITHMS)}"
            ) from exc

        self.algorithm = algorithm
        self._model: DigitClassificationInterface = model_cls()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.train(X, y)

    def predict(self, image: np.ndarray) -> int:
        image = np.asarray(image)
        if image.shape != CANONICAL_INPUT_SHAPE:
            raise ValueError(
                f"DigitClassifier.predict expects a {CANONICAL_INPUT_SHAPE} image, got {image.shape}"
            )

        adapted = self._adapt(image)
        return self._model.predict(adapted)

    def _adapt(self, image: np.ndarray) -> np.ndarray:
        if self.algorithm == "cnn":
            return image
        if self.algorithm == "rf":
            return image.reshape(784)
        if self.algorithm == "rand":
            return _center_crop(image, 10, 10)
        raise AssertionError(f"unreachable algorithm: {self.algorithm}")


def _center_crop(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    height, width, _ = image.shape
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[top : top + crop_h, left : left + crop_w, 0]
