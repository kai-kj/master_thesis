# %%

import torch
from typing import Literal
import segmentation_models_pytorch as smp
from pathlib import Path

from scripts.models.model import Model


# %%


class DeepLabV3Plus(Model):
    def __init__(
            self,
            n_classes: int,
            variant: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet34",
            apply_softmax: bool = False,
            get_weights: bool = True,
            **kwargs
    ):
        super().__init__()

        self._name = "DeepLabV3Plus"
        self._config = {
            "n_classes": n_classes,
            "variant": variant,
            "apply_softmax": apply_softmax,
        }

        self.model = smp.DeepLabV3Plus(
            encoder_name=variant,
            encoder_weights="imagenet" if get_weights else None,
            in_channels=3,
            classes=n_classes,
        )

        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "DeepLabV3Plus":
        deeplab = Model.load(config_path, weights_path)
        assert isinstance(deeplab, DeepLabV3Plus)
        return deeplab

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.model.encoder]

    def get_decoder_modules(self) -> list[torch.nn.Module]:
        return [self.model.decoder]

    def get_head_modules(self) -> list[torch.nn.Module]:
        return [self.model.segmentation_head]

    def get_encoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_encoder_modules() for p in m.parameters()]

    def get_decoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_decoder_modules() for p in m.parameters()]

    def get_head_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_head_modules() for p in m.parameters()]

    def change_n_classes(self, n_classes: int):
        head = self.model.segmentation_head
        device = head[0].weight.device
        dtype = head[0].weight.dtype

        head[0] = torch.nn.Conv2d(
            in_channels=head[0].in_channels,  # type: ignore
            out_channels=n_classes,
            kernel_size=head[0].kernel_size,  # type: ignore
            stride=head[0].stride,  # type: ignore
        ).to(device=device, dtype=dtype)  # type: ignore

        self._config["n_classes"] = n_classes


# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = DeepLabV3Plus(len(dataset.label_id_to_name))
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")

    print(f"before change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")

    model.change_n_classes(8)

    print(f"after change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")
