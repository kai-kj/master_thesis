# %%

import torch
from typing import Literal
from pathlib import Path
import json

from scripts.models.model import Model


# %%

class ConvReLU(torch.nn.Module):
    def __init__(
            self,
            in_ch: int, out_ch: int,
            kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1,
            norm: Literal["none", "instance", "group"] = "none"
    ):
        super().__init__()

        match norm:
            case "none":
                self.seq = torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation),
                    torch.nn.ReLU(inplace=True),
                )
            case "instance":
                self.seq = torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation),
                    torch.nn.InstanceNorm2d(out_ch, affine=True),
                    torch.nn.ReLU(inplace=True),
                )
            case "group":
                self.seq = torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation),
                    torch.nn.GroupNorm(self.get_num_groups(in_ch), out_ch),
                    torch.nn.ReLU(inplace=True),
                )

    def forward(self, x):
        return self.seq(x)

    def get_num_groups(self, in_ch: int) -> int:
        for g in [8, 4, 2]:
            if in_ch % g == 0:
                return g

        return 1


class UNet(Model):
    def __init__(
            self,
            n_classes: int,
            apply_softmax: bool = False,
            norm: Literal["none", "instance", "group"] = "none",
            **kwargs
    ):
        super().__init__()

        self._config = {"n_classes": n_classes, "apply_softmax": apply_softmax, "norm": norm}

        # in -> 1024x1024x3
        self.conv11 = ConvReLU(3, 64, norm=norm)  # -> 1024x1024x64
        self.conv12 = ConvReLU(64, 64, norm=norm)  # -> 1024x1024x64
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512x512x64

        self.conv21 = ConvReLU(64, 128, norm=norm)  # -> 512x512x128
        self.conv22 = ConvReLU(128, 128, norm=norm)  # -> 512x512x128
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 256x256x128

        self.conv31 = ConvReLU(128, 256, norm=norm)  # -> 256x256x256
        self.conv32 = ConvReLU(256, 256, norm=norm)  # -> 256x256x256
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 128x128x256

        self.conv41 = ConvReLU(256, 512, norm=norm)  # -> 128x128x512
        self.conv42 = ConvReLU(512, 512, norm=norm)  # -> 128x128x512
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 64x64x512

        self.conv51 = ConvReLU(512, 1024, norm=norm)  # -> 64x64x1024
        self.conv52 = ConvReLU(1024, 1024, norm=norm)  # -> 64x64x1024
        self.up5 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # -> 128x128x512

        # (cat -> 128x128x1024)
        self.conv61 = ConvReLU(1024, 512, norm=norm)  # -> 128x128x512
        self.conv62 = ConvReLU(512, 512, norm=norm)  # -> 128x128x512
        self.up6 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # -> 256x256x256

        # (cat -> 256x256x512)
        self.conv71 = ConvReLU(512, 256, norm=norm)  # -> 256x256x256
        self.conv72 = ConvReLU(256, 256, norm=norm)  # -> 256x256x256
        self.up7 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # -> 512x512x128

        # (cat -> 512x512x256)
        self.conv81 = ConvReLU(256, 128, norm=norm)  # -> 512x512x128
        self.conv82 = ConvReLU(128, 128, norm=norm)  # -> 512x512x128
        self.up8 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # -> 1024x1024x64

        # (cat -> 1024x1024x128)
        self.conv91 = ConvReLU(128, 64, norm=norm)  # -> 1024x1024x64
        self.conv92 = ConvReLU(64, 64, norm=norm)  # -> 1024x1024x64
        self.conv93 = torch.nn.ConvTranspose2d(64, n_classes, kernel_size=1)  # -> 1024x1024xn_classes

        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.conv11(x)
        x13 = self.conv12(x12)
        x21 = self.pool1(x13)

        x22 = self.conv21(x21)
        x23 = self.conv22(x22)
        x31 = self.pool2(x23)

        x32 = self.conv31(x31)
        x33 = self.conv32(x32)
        x41 = self.pool3(x33)

        x42 = self.conv41(x41)
        x43 = self.conv42(x42)
        x51 = self.pool4(x43)

        x52 = self.conv51(x51)
        x53 = self.conv52(x52)
        x61 = torch.cat((self.up5(x53), x43), dim=1)

        x62 = self.conv61(x61)
        x63 = self.conv62(x62)
        x71 = torch.cat((self.up6(x63), x33), dim=1)

        x72 = self.conv71(x71)
        x73 = self.conv72(x72)
        x81 = torch.cat((self.up7(x73), x23), dim=1)

        x82 = self.conv81(x81)
        x83 = self.conv82(x82)
        x91 = torch.cat((self.up8(x83), x13), dim=1)

        x92 = self.conv91(x91)
        x93 = self.conv92(x92)
        x101 = self.conv93(x93)

        return torch.nn.functional.softmax(x101, dim=1) if self.apply_softmax else x101

    @staticmethod
    def load(config_path: str | Path, weights_path: str | Path) -> "UNet":
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.loads(config_path.read_text())
        weights = torch.load(weights_path, weights_only=True)

        assert config["name"] == "UNet"

        unet = UNet(**config)
        unet.load_state_dict(weights)

        return unet

    def save(self, config_path: str | Path, weights_path: str | Path):
        config_path = Path(config_path)
        weights_path = Path(weights_path)

        config = json.dumps({**self._config, "name": "UNet"}, indent=2)
        weights = self.state_dict()

        config_path.write_text(config)
        torch.save(weights, weights_path)

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [
            self.conv11, self.conv12,
            self.conv21, self.conv22,
            self.conv31, self.conv32,
            self.conv41, self.conv42,
        ]

    def get_decoder_modules(self) -> list[torch.nn.Module]:
        return [
            self.up5, self.conv61, self.conv62,
            self.up6, self.conv71, self.conv72,
            self.up7, self.conv81, self.conv82,
            self.up8, self.conv91, self.conv92,
            self.conv93,
        ]

    def change_n_classes(self, n_classes: int):
        in_channels = self.conv93.in_channels
        device = next(self.conv93.parameters()).device
        dtype = next(self.conv93.parameters()).dtype

        self.conv93 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=n_classes,
            kernel_size=1
        ).to(device=device, dtype=dtype)

        self._config["n_classes"] = n_classes


# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(len(dataset.label_id_to_name), norm="none")
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")

    # torch.onnx.export(
    #     model,
    #     (next(iter(dataloader))[0],),
    #     f"models/untrained_onnx/unet.onnx",
    #     input_names=["input"],
    #     output_names=["output"],
    #     external_data=False,
    #     dynamo=True,
    # )
