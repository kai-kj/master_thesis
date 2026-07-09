import torch
from pathlib import Path
import json

from scripts.models.unet_res import UNetRes


class UnetFusionPredictor(torch.nn.Module):
    def __init__(self, unet: UNetRes | None, n_outputs: int, hidden_dim: int = 512, freeze_encoder: bool = False):
        super().__init__()

        self._config = {
            "n_outputs": n_outputs,
            "hidden_dim": hidden_dim,
            "freeze_encoder": freeze_encoder,
        }

        self.freeze_encoder = freeze_encoder

        self.encoder = unet.model.encoder if unet is not None else UNetRes(10, "resnet34").model.encoder
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.out_channels[-1], hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = torch.flatten(self.pool(features[-1]), 1)
        return self.head(pooled)

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_encoder:
            self.encoder.eval()

        return self

    @staticmethod
    def load(config_path: Path, weights_path: Path) -> "UnetFusionPredictor":
        config = json.loads(config_path.read_text())
        weights = torch.load(weights_path, weights_only=True)

        model = UnetFusionPredictor(**{**config, "unet": None})
        model.load_state_dict(weights)

        return model

    def save(self, config_path: Path, weights_path: Path):
        config_path.write_text(json.dumps(self._config, indent=2))
        torch.save(self.state_dict(), weights_path)
