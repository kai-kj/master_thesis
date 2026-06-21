import torch
from pathlib import Path
import json


class Model(torch.nn.Module):
    def __init__(self):
        self._name = None
        self._config = dict()
        super().__init__()

    def forward(self, x: torch.Tensor, x_fusion: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "Model":
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.loads(config_path.read_text())
        weights = torch.load(weights_path, weights_only=True)

        if "fusion" not in config:
            config["fusion"] = None

        match config["name"]:
            case "UNetRes":
                from scripts.models.unet_res import UNetRes
                model = UNetRes(**{**config, "get_weights": False})
                model.load_state_dict(weights)
                return model

            case "DeepLabV3Plus":
                from scripts.models.deeplab import DeepLabV3Plus
                model = DeepLabV3Plus(**{**config, "get_weights": False})
                model.load_state_dict(weights)
                return model

            case "SegFormer":
                from scripts.models.segformer import SegFormer
                model = SegFormer(**{**config, "get_weights": False})
                model.load_state_dict(weights)
                return model

            case "TransUNet":
                from scripts.models.transunet import TransUNet
                model = TransUNet(**{**config, "get_weights": False})
                model.load_state_dict(weights)
                return model

            case "SwinFPN":
                from scripts.models.swin_fpn import SwinFPN
                model = SwinFPN(**{**config, "get_weights": False})
                model.load_state_dict(weights)
                return model

        raise RuntimeError(f"{config["name"]} not implemented")

    def save(self, config_path: str | Path, weights_path: str | Path) -> None:
        assert self._name is not None

        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.dumps({**self._config, "name": self._name}, indent=2)
        weights = self.state_dict()

        config_path.write_text(config)
        torch.save(weights, weights_path)

    def change_n_classes(self, n_classes: int):
        raise NotImplementedError

    def count_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) \
            if trainable_only \
            else sum(p.numel() for p in self.parameters())

    def export(self, path: str | Path, x: torch.Tensor) -> None:
        torch.onnx.export(
            self,
            (x,),
            path,
            input_names=["input"],
            output_names=["output"],
            external_data=False,
            dynamo=True,
        )
