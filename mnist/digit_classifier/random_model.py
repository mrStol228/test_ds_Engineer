from typing import Optional

import numpy as np

from .interface import DigitClassificationInterface


class RandomModel(DigitClassificationInterface):
    INPUT_SHAPE = (10, 10)

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Training is out of scope for this exercise.")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != self.INPUT_SHAPE:
            raise ValueError(f"RandomModel expects a {self.INPUT_SHAPE} array, got {image.shape}")

        return int(self._rng.integers(0, 10))
