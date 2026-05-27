# %%

import torch
from typing import Literal
from pathlib import Path
import json
import einops
import numpy as np

from scripts.models.model import Model


# %%

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, head_count):
        super().__init__()

        self.head_count = head_count
        self.dk = (embedding_size // head_count) ** (1 / 2)

        self.qkv_layer = torch.nn.Linear(embedding_size, embedding_size * 3, bias=False)
        self.out_attention = torch.nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, x0, mask=None):
        qkv = self.qkv_layer(x0)

        query, key, value = tuple(einops.rearrange(qkv, "b t (d k h ) -> k b h t d ", k=3, h=self.head_count))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x1 = torch.einsum("... i j , ... j d -> ... i d", attention, value)
        x2 = einops.rearrange(x1, "b h t d -> b t (h d)")

        return self.out_attention(x2)


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embedding_size: int, head_count: int, mlp_size: int):
        super().__init__()

        self.mha = MultiHeadAttention(embedding_size, head_count)

        self.drop = torch.nn.Dropout(0.1)

        self.norm1 = torch.nn.LayerNorm(embedding_size)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, mlp_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(mlp_size, embedding_size),
            torch.nn.Dropout(0.1)
        )

        self.norm2 = torch.nn.LayerNorm(embedding_size)

    def forward(self, x0):
        x1 = self.mha(x0)
        x2 = self.drop(x1)
        x3 = x0 + x2
        x4 = self.norm1(x3)

        x5 = self.mlp(x4)
        x6 = x4 + x5

        return self.norm2(x6)


class ViT(torch.nn.Module):
    def __init__(
            self,
            image_size: int,
            in_channel_count: int,
            embedding_size: int,
            head_count: int,
            mlp_size: int,
            block_count: int,
            patch_size: int
    ):
        super().__init__()

        self.patch_size = patch_size
        self.token_count = (image_size // patch_size) ** 2
        self.token_size = in_channel_count * (patch_size ** 2)

        self.project = torch.nn.Linear(self.token_size, embedding_size)
        self.embedd = torch.nn.Parameter(torch.rand(self.token_count + 1, embedding_size))

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embedding_size))

        self.dropout = torch.nn.Dropout(0.1)

        self.transformer = torch.nn.ModuleList([
            TransformerEncoderBlock(embedding_size, head_count, mlp_size) for _ in range(block_count)
        ])

    def forward(self, x0):
        image_patches = einops.rearrange(
            x0,
            "b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)",
            patch_x=self.patch_size, patch_y=self.patch_size
        )

        batch_size, tokens, _ = image_patches.shape

        projected = self.project(image_patches)

        token = einops.repeat(
            self.cls_token,
            "b ... -> (b batch_size) ...",
            batch_size=batch_size
        )

        patches = torch.cat([token, projected], dim=1)
        patches += self.embedd[:tokens + 1, :]

        x1 = self.dropout(patches)

        for layer_block in self.transformer:
            x1 = layer_block(x1)

        return x1[:, 1:, :]


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, base_width: int = 64):
        super().__init__()

        width = int(out_channels * (base_width / 64))

        self.conv_down = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.norm_down = torch.nn.BatchNorm2d(out_channels)

        self.conv1 = torch.nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = torch.nn.BatchNorm2d(width)

        self.conv2 = torch.nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(width)

        self.conv3 = torch.nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = torch.nn.functional.relu(self.norm1(self.conv1(x)), inplace=True)
        x2 = torch.nn.functional.relu(self.norm2(self.conv2(x1)), inplace=True)
        x3 = torch.nn.functional.relu(self.norm3(self.conv3(x2)) + self.norm_down(self.conv_down(x)), inplace=True)
        return x3


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channel_count: int, embedding_size: int, scale_factor: int = 2):
        super().__init__()

        self.up = torch.nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        self.conv1 = torch.nn.Conv2d(in_channel_count, embedding_size, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(embedding_size)

        self.conv2 = torch.nn.Conv2d(embedding_size, embedding_size, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(embedding_size)

    def forward(self, x: torch.Tensor, x_concat: torch.Tensor | None = None):
        x0 = self.up(x)

        if x_concat is not None:
            x0 = torch.cat([x_concat, x0], dim=1)

        x1 = torch.nn.functional.relu(self.norm1(self.conv1(x0)), inplace=True)
        x2 = torch.nn.functional.relu(self.norm2(self.conv2(x1)), inplace=True)

        return x2


class TransUNet(Model):
    def __init__(
            self,
            class_count: int,
            image_size: int = 512,
            in_channel_count: int = 3,
            embedding_size: int = 128,
            head_count: int = 4,
            mlp_size: int = 512,
            block_count: int = 8,
            patch_size: int = 16,
            **kwargs
    ):
        super().__init__()

        self._config = {
            "class_count": class_count,
            "image_size": image_size,
            "in_channel_count": in_channel_count,
            "embedding_size": embedding_size,
            "head_count": head_count,
            "mlp_size": mlp_size,
            "block_count": block_count,
            "patch_size": patch_size,
        }

        self.encoder_begin = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel_count, embedding_size, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(embedding_size),
            torch.nn.ReLU(inplace=True),
        )

        self.encoder1 = EncoderBlock(embedding_size, embedding_size * 2, stride=2)
        self.encoder2 = EncoderBlock(embedding_size * 2, embedding_size * 4, stride=2)
        self.encoder3 = EncoderBlock(embedding_size * 4, embedding_size * 8, stride=2)

        self.vit_img_dim = image_size // patch_size
        self.vit = ViT(
            self.vit_img_dim, embedding_size * 8, embedding_size * 8,
            head_count, mlp_size, block_count, patch_size=1
        )

        self.encoder_end = torch.nn.Sequential(
            torch.nn.Conv2d(embedding_size * 8, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )

        self.decoder1 = DecoderBlock(embedding_size * 8, embedding_size * 2)
        self.decoder2 = DecoderBlock(embedding_size * 4, embedding_size)
        self.decoder3 = DecoderBlock(embedding_size * 2, int(embedding_size * 1 / 2))
        self.decoder4 = DecoderBlock(int(embedding_size * 1 / 2), int(embedding_size * 1 / 8))

        self.decoder_end = torch.nn.Conv2d(int(embedding_size * 1 / 8), class_count, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder_begin(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        x5 = self.vit(x4)
        x6 = einops.rearrange(x5, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x7 = self.encoder_end(x6)

        x8 = self.decoder1(x7, x3)
        x9 = self.decoder2(x8, x2)
        x10 = self.decoder3(x9, x1)
        x11 = self.decoder4(x10)

        x12 = self.decoder_end(x11)

        return x12

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "TransUNet":
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.loads(config_path.read_text())
        weights = torch.load(weights_path, weights_only=True)

        assert config["name"] == "TransUNet"

        unet = TransUNet(**config)
        unet.load_state_dict(weights)

        return unet

    def save(self, config_path: str | Path, weights_path: str | Path):
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.dumps({**self._config, "name": "TransUNet"}, indent=2)
        weights = self.state_dict()

        config_path.write_text(config)
        torch.save(weights, weights_path)

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.encoder_begin, self.encoder1, self.encoder2, self.encoder3]

    def get_decoder_modules(self) -> list[torch.nn.Module]:
        return [self.encoder_end, self.decoder1, self.decoder2, self.decoder3, self.decoder_end]

    def get_transformer_modules(self) -> list[torch.nn.Module]:
        return [self.vit]

    def get_encoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_encoder_modules() for p in m.parameters()]

    def get_decoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_decoder_modules() for p in m.parameters()]

    def get_transformer_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_transformer_modules() for p in m.parameters()]

        # %%


if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = TransUNet(len(dataset.label_id_to_name), image_size=512, in_channel_count=3)
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")

    print(f"in  shape: {next(iter(dataloader))[0].shape}")
    print(f"out shape: {model(next(iter(dataloader))[0]).shape}")

    torch.onnx.export(
        model,
        (next(iter(dataloader))[0],),
        f"models/untrained_onnx/transunet.onnx",
        input_names=["input"],
        output_names=["output"],
        external_data=False,
        dynamo=True,
    )
