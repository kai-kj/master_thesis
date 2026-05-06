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

    def count_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) \
            if trainable_only \
            else sum(p.numel() for p in self.parameters())
