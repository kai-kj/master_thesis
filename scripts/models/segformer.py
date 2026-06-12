# %%

from typing import Literal
import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig, logging as transformer_logging
from pathlib import Path

from scripts.models.model import Model

transformer_logging.set_verbosity_error()


# %%

class SegFormer(Model):
    VARIANTS = {
        "b0": "nvidia/mit-b0",
        "b1": "nvidia/mit-b1",
        "b2": "nvidia/mit-b2",
        "b3": "nvidia/mit-b3",
        "b4": "nvidia/mit-b4",
        "b5": "nvidia/mit-b5",
    }

    def __init__(
            self,
            n_classes: int,
            variant: Literal["b0", "b1", "b2", "b3", "b4", "b5"] = "b2",
            apply_softmax: bool = False,
            get_weights: bool = True,
            **kwargs
    ):
        super().__init__()

        self._name = "SegFormer"
        self._config = {
            "n_classes": n_classes,
            "variant": variant,
            "apply_softmax": apply_softmax,
        }

        if get_weights:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                SegFormer.VARIANTS[variant],
                num_labels=n_classes, ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig.from_pretrained(SegFormer.VARIANTS[variant])
            cfg.num_labels = n_classes
            self.model = SegformerForSemanticSegmentation(cfg)

        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)

        logits = torch.nn.functional.interpolate(
            out.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "SegFormer":
        segformer = Model.load(config_path, weights_path)
        assert isinstance(segformer, SegFormer)
        return segformer

    def change_n_classes(self, n_classes: int):
        head = self.model.decode_head  # type: ignore
        device = head.classifier.weight.device
        dtype = head.classifier.weight.dtype

        head.classifier = torch.nn.Conv2d(
            in_channels=head.classifier.in_channels,
            out_channels=n_classes,
            kernel_size=1
        ).to(device=device, dtype=dtype)

        self._config["n_classes"] = n_classes

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.model.segformer]  # type: ignore

    def get_decoder_modules(self) -> list[torch.nn.Module]:
        return [self.model.decode_head]  # type: ignore

    def get_encoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_encoder_modules() for p in m.parameters()]

    def get_decoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_decoder_modules() for p in m.parameters()]


# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = SegFormer(len(dataset.label_id_to_name), "b2")
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")

    print(f"before change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")

    model.change_n_classes(8)

    print(f"after change_n_classes,")
    print(f"- in  shape: {next(iter(dataloader))[0].shape}")
    print(f"- out shape: {model(next(iter(dataloader))[0]).shape}")
