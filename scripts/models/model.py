import torch
from pathlib import Path


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "Model":
        raise NotImplementedError

    def save(self, config_path: str | Path, weights_path: str | Path) -> None:
        raise NotImplementedError
