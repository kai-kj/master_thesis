# %%

import torch
from typing import Literal
import segmentation_models_pytorch as smp
from pathlib import Path

from scripts.models.model import Model
from scripts.models.fusion import FusionModule, FusionFILM, FusionConcat, FusionLogit
from dependencies.clim_soil_encoder.model import FusionEncoder, MLPEncoder, MLPModel


class UNetRes(Model):
    def __init__(
            self,
            n_classes: int,
            variant: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet34",
            apply_softmax: bool = False,
            get_weights: bool = True,
            fusion_mode: None | Literal["concat", "film", "logit"] = None,
            fusion_encoder: FusionEncoder | None = None,
            **kwargs
    ):
        super().__init__()

        self._name = "UNetRes"
        self._config = {
            "n_classes": n_classes,
            "variant": variant,
            "apply_softmax": apply_softmax,
            "fusion_mode": fusion_mode,
        }

        self.model = smp.Unet(
            encoder_name=variant,
            encoder_weights="imagenet" if get_weights else None,
            in_channels=3,
            classes=n_classes,
        )

        self.apply_softmax = apply_softmax

        self.fusion_encoder: FusionEncoder | None = fusion_encoder
        self.fusion_module: FusionModule | None = None
        self.fusion_mode = fusion_mode

        match self.fusion_mode:
            case "concat":
                assert self.fusion_encoder is not None
                self.fusion_module = FusionConcat(
                    self.model.encoder.out_channels[-1],
                    self.fusion_encoder.get_n_out_features()
                )

            case "film":
                assert self.fusion_encoder is not None
                self.fusion_module = FusionFILM(
                    self.model.encoder.out_channels[-1],
                    self.fusion_encoder.get_n_out_features()
                )

            case "logit":
                assert self.fusion_encoder is not None
                self.fusion_module = FusionLogit(n_classes, 8)

    def forward(self, x: torch.Tensor, x_fusion: torch.Tensor | None = None) -> torch.Tensor:
        match self.fusion_mode:
            case "concat" | "film":
                assert self.fusion_encoder is not None
                assert self.fusion_module is not None
                assert x_fusion is not None

                image_features = self.model.encoder(x)
                fusion_features = self.fusion_encoder(x_fusion)

                image_features[-1] = self.fusion_module(image_features[-1], fusion_features)
                logits = self.model.segmentation_head(self.model.decoder(image_features))

                return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

            case "logit":
                assert self.fusion_encoder is not None
                assert self.fusion_module is not None
                assert x_fusion is not None

                image_features = self.model(x)
                fusion_features = self.fusion_encoder(x_fusion)

                logits = self.fusion_module(image_features, fusion_features)

                return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

            case _:
                logits = self.model(x)
                return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "UNetRes":
        unet = Model.load(config_path, weights_path)
        assert unet._name == "UNetRes"
        return unet  # type: ignore

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.model.encoder]

    def get_decoder_modules(self) -> list[torch.nn.Module]:
        return [self.model.decoder]

    def get_head_modules(self) -> list[torch.nn.Module]:
        return [self.model.segmentation_head]

    def get_fusion_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.fusion_encoder] if self.fusion_encoder is not None else []

    def get_fusion_modules(self) -> list[torch.nn.Module]:
        return [self.fusion_module] if self.fusion_module is not None else []

    def get_encoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_encoder_modules() for p in m.parameters()]

    def get_decoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_decoder_modules() for p in m.parameters()]

    def get_head_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_head_modules() for p in m.parameters()]

    def get_fusion_encoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_fusion_encoder_modules() for p in m.parameters()]

    def get_fusion_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_fusion_modules() for p in m.parameters()]

    def change_n_classes(self, n_classes: int):
        head = self.model.segmentation_head
        device = head[0].weight.device
        dtype = head[0].weight.dtype

        head[0] = torch.nn.Conv2d(
            in_channels=head[0].in_channels,  # type: ignore
            out_channels=n_classes,
            kernel_size=head[0].kernel_size,  # type: ignore
            stride=head[0].stride,  # type: ignore
            padding=head[0].padding,  # type: ignore
        ).to(device=device, dtype=dtype)  # type: ignore

        if self.fusion_mode == "logit":
            self.fusion_module = FusionLogit(n_classes, 8)

        self._config["n_classes"] = n_classes


# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNetRes(len(dataset.label_id_to_name))
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")

    print(f"before change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")

    model.change_n_classes(8)

    print(f"after change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")

# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI
    from dependencies.clim_soil_encoder.dataset import DatasetClimateSoil

    dataset_image = DatasetTreeAI("ID", "train", 0)
    dataloader_image = torch.utils.data.DataLoader(dataset_image, batch_size=1, shuffle=True)

    dataset_fusion = DatasetClimateSoil("train", 0, include_clim=True)
    dataloader_fusion = torch.utils.data.DataLoader(dataset_fusion, batch_size=1, shuffle=True)

    n_classes = len(dataset_image.label_id_to_name)

    # model = UNetRes(n_classes, fusion_mode="concat", fusion_encoder=MLPEncoder(8, 64))
    # model = UNetRes(n_classes, fusion_mode="film", fusion_encoder=MLPEncoder(8, 64))
    model = UNetRes(n_classes, fusion_mode="logit", fusion_encoder=MLPModel(8, 8, 64, apply_softmax=False))

    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")

    print(f"before change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader_image))[0].shape}, {next(iter(dataloader_fusion))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader_image))[0], next(iter(dataloader_fusion))[0]).shape}")

    model.save("tmp/test.json", "tmp/test.pt")

    model.change_n_classes(8)

    print(f"after change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader_image))[0].shape}, {next(iter(dataloader_fusion))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader_image))[0], next(iter(dataloader_fusion))[0]).shape}")

# %%

if __name__ == "__main__":
    model = UNetRes.load(
        "tmp/test.json",
        "tmp/test.pt"
    )

    # model = UNetRes.load("tmp/unet.json", "tmp/unet.pt")

    pass
