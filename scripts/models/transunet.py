# %%

import torch
from typing import Literal
import numpy as np
from pathlib import Path

from dependencies.TransUNet.networks.vit_seg_modeling import VisionTransformer as transunet_model
from dependencies.TransUNet.networks.vit_seg_modeling import CONFIGS as transunet_config
from scripts.models.model import Model


# %%

class TransUNet(Model):
    VARIANTS = {
        "b16": "ViT-B_16",
        "b32": "ViT-B_32",
        "l16": "ViT-L_16",
        "l32": "ViT-L_32",
        "h14": "ViT-H_14",
    }

    WEIGHTS_PATH = Path("/home/kai/Thesis/model_weights/")

    def __init__(
            self,
            n_classes: int,
            variant: Literal["b16", "b32", "l16", "l32", "h14"] = "b16",
            apply_softmax: bool = False,
            get_weights: bool = True,
            **kwargs
    ):
        super().__init__()

        self._name = "TransUNet"
        self._config = {
            "n_classes": n_classes,
            "variant": variant,
            "apply_softmax": apply_softmax,
        }

        config = transunet_config[TransUNet.VARIANTS[variant]].copy_and_resolve_references()
        config.n_classes = n_classes
        config.n_skip = 0

        self.model = transunet_model(config, img_size=512, num_classes=config.n_classes)

        if get_weights:
            self.model.load_from(np.load(TransUNet.WEIGHTS_PATH / f"imagenet21k_{TransUNet.VARIANTS[variant]}.npz"))

        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)

        return torch.nn.functional.softmax(out, dim=1) if self.apply_softmax else out

    def count_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) \
            if trainable_only \
            else sum(p.numel() for p in self.parameters())

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.model.transformer]

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


# %%


if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = TransUNet(len(dataset.label_id_to_name), variant="b16")
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")
    print(f"in  shape: {next(iter(dataloader))[0].shape}")
    print(f"out shape: {model(next(iter(dataloader))[0]).shape}")
