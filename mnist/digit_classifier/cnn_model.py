import numpy as np
import torch
from torch import nn

from .interface import DigitClassificationInterface


class _CnnNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class CnnModel(DigitClassificationInterface):
    INPUT_SHAPE = (28, 28, 1)

    def __init__(self) -> None:
        self._net = _CnnNet()
        self._net.eval()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Training is out of scope for this exercise.")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != self.INPUT_SHAPE:
            raise ValueError(f"CnnModel expects a {self.INPUT_SHAPE} image, got {image.shape}")

        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = self._net(tensor)
        return int(torch.argmax(logits, dim=1).item())
