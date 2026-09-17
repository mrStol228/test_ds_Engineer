from abc import ABC, abstractmethod

import numpy as np


class DigitClassificationInterface(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        raise NotImplementedError
