import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .interface import DigitClassificationInterface


class RandomForestModel(DigitClassificationInterface):
    INPUT_SHAPE = (784,)

    def __init__(self) -> None:
        self._model = RandomForestClassifier(n_estimators=10, random_state=42)
        self._warm_start()

    def _warm_start(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.random((20, *self.INPUT_SHAPE))
        y = np.tile(np.arange(10), 2)
        self._model.fit(X, y)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Training is out of scope for this exercise.")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != self.INPUT_SHAPE:
            raise ValueError(f"RandomForestModel expects a {self.INPUT_SHAPE} array, got {image.shape}")

        return int(self._model.predict(image.reshape(1, -1))[0])
