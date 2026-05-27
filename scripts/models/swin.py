# %%

import json
from pathlib import Path
from typing import Literal

import torch
from transformers import UperNetForSemanticSegmentation, UperNetConfig, logging as transformer_logging

from scripts.models.model import Model

transformer_logging.set_verbosity_error()


# %%

class SwinUPerNet(Model):
    VARIANTS = {
        "s": "openmmlab/upernet-swin-small",
        "b": "openmmlab/upernet-swin-base",
        "l": "openmmlab/upernet-swin-large",
    }

    def __init__(
            self,
            n_classes: int,
            variant: Literal["s", "b", "l"] = "b",
            apply_softmax: bool = False,
            get_weights: bool = True,
            **kwargs,
    ):
        super().__init__()

        self._config = {
            "n_classes": n_classes,
            "variant": variant,
            "apply_softmax": apply_softmax,
        }

        self.apply_softmax = apply_softmax

        if get_weights:
            self.model = UperNetForSemanticSegmentation.from_pretrained(
                SwinUPerNet.VARIANTS[variant],
                num_labels=n_classes, ignore_mismatched_sizes=True,
            )
        else:
            cfg = UperNetConfig.from_pretrained(SwinUPerNet.VARIANTS[variant])
            cfg.num_labels = n_classes
            self.model = UperNetForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)

        logits = torch.nn.functional.interpolate(
            out.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

    def get_backbone_modules(self) -> list[torch.nn.Module]:
        return [self.model.backbone]  # type: ignore

    def get_dhead_modules(self) -> list[torch.nn.Module]:
        return [self.model.decode_head]  # type: ignore

    def get_ahead_modules(self) -> list[torch.nn.Module]:
        return [self.model.auxiliary_head]  # type: ignore

    def get_backbone_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_backbone_modules() for p in m.parameters()]

    def get_dhead_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_dhead_modules() for p in m.parameters()]

    def get_ahead_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_ahead_modules() for p in m.parameters()]

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "SwinUPerNet":
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.loads(config_path.read_text())
        weights = torch.load(weights_path, weights_only=True)

        assert config["name"] == "SwinUPerNet"

        swin = SwinUPerNet(**{**config, "get_weights": False})
        swin.load_state_dict(weights)

        return swin

    def save(self, config_path: str | Path, weights_path: str | Path):
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.dumps({**self._config, "name": "SwinUPerNet"}, indent=2)
        weights = self.state_dict()

        config_path.write_text(config)
        torch.save(weights, weights_path)


# %%


if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = SwinUPerNet(len(dataset.label_id_to_name), "b")
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")
    print(f"in  shape: {next(iter(dataloader))[0].shape}")
    print(f"out shape: {model(next(iter(dataloader))[0]).shape}")

    # torch.onnx.export(
    #     model,
    #     (next(iter(dataloader))[0],),
    #     f"models/untrained_onnx/segformer.onnx",
    #     input_names=["input"],
    #     output_names=["output"],
    #     external_data=False,
    #     dynamo=True,
    # )
