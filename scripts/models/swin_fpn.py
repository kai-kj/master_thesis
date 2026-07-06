# %%

from typing import Literal
import torch
from transformers import SwinModel, SwinConfig, logging as transformer_logging
from pathlib import Path

from scripts.models.model import Model

transformer_logging.set_verbosity_error()


# %%

class FPN(torch.nn.Module):
    def __init__(self, in_channels: list[int], fpn_dim: int = 256):
        super().__init__()

        self.lat_conv = torch.nn.ModuleList([
            torch.nn.Conv2d(c, fpn_dim, kernel_size=1)
            for c in in_channels
        ])

        self.out_conv = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(fpn_dim),
                torch.nn.ReLU(inplace=True),
            )
            for _ in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        features = [conv(f) for conv, f in zip(self.lat_conv, features)]

        for i in range(len(features) - 1, 0, -1):
            features[i - 1] = features[i - 1] + torch.nn.functional.interpolate(
                features[i],
                size=features[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        features = [conv(f) for conv, f in zip(self.out_conv, features)]

        return features[0]


class SwinFPN(Model):
    VARIANTS = {
        "s": "microsoft/swin-small-patch4-window7-224",
        "b": "microsoft/swin-base-patch4-window7-224",
        "l": "microsoft/swin-large-patch4-window7-224",
    }

    HIDDEN_SIZES = {
        "s": [96, 192, 384, 768],
        "b": [128, 256, 512, 1024],
        "l": [192, 384, 768, 1536],
    }

    def __init__(
            self,
            n_classes: int,
            variant: Literal["s", "b", "l"] = "b",
            apply_softmax: bool = False,
            get_weights: bool = True,
            fpn_dim: int = 256,
            **kwargs,
    ):
        super().__init__()

        self._name = "SwinFPN"

        self._config = {
            "n_classes": n_classes,
            "variant": variant,
            "apply_softmax": apply_softmax,
            "fpn_dim": fpn_dim,
        }

        self.apply_softmax = apply_softmax

        if get_weights:
            self.encoder = SwinModel.from_pretrained(SwinFPN.VARIANTS[variant], output_hidden_states=True)
        else:
            cfg = SwinConfig.from_pretrained(SwinFPN.VARIANTS[variant])
            cfg.output_hidden_states = True
            self.encoder = SwinModel(cfg)

        self.decoder = FPN(in_channels=SwinFPN.HIDDEN_SIZES[variant], fpn_dim=fpn_dim)
        self.head = torch.nn.Conv2d(fpn_dim, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, x_fusion: torch.Tensor | None = None) -> torch.Tensor:
        features = list(self.encoder(pixel_values=x, output_hidden_states=True).reshaped_hidden_states)
        logits = self.head(self.decoder(features))

        logits = torch.nn.functional.interpolate(
            logits, size=(x.shape[2], x.shape[3]),
            mode="bilinear", align_corners=False,
        )

        return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "SwinFPN":
        swin = Model.load(config_path, weights_path)
        assert isinstance(swin, SwinFPN)
        return swin

    def change_n_classes(self, n_classes: int):
        device = self.head.weight.device
        dtype = self.head.weight.dtype

        self.head = torch.nn.Conv2d(self.head.in_channels, n_classes, kernel_size=1).to(device=device, dtype=dtype)
        self._config["n_classes"] = n_classes

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.encoder]

    def get_decoder_modules(self) -> list[torch.nn.Module]:
        return [self.decoder]

    def get_head_modules(self) -> list[torch.nn.Module]:
        return [self.head]

    def get_encoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_encoder_modules() for p in m.parameters()]

    def get_decoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_decoder_modules() for p in m.parameters()]

    def get_head_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_head_modules() for p in m.parameters()]


# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = SwinFPN(len(dataset.label_id_to_name), "b")
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")

    print(f"before change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")

    model.change_n_classes(8)

    print(f"after change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")
