# %%

import torch
import torchvision

from scripts.models.model import Model


# %%

class VGG16Encoder(torch.nn.Module):
    def __init__(self, get_weights: bool = True):
        super().__init__()

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features \
            if get_weights \
            else torchvision.models.vgg16().features

        self.encoder1 = torch.nn.Sequential(*vgg[:4])  # type: ignore -> 64
        self.encoder2 = torch.nn.Sequential(*vgg[4:9])  # type: ignore -> 128
        self.encoder3 = torch.nn.Sequential(*vgg[9:16])  # type: ignore -> 256
        self.encoder4 = torch.nn.Sequential(*vgg[16:23])  # type: ignore -> 512
        self.encoder5 = torch.nn.Sequential(*vgg[23:30])  # type: ignore -> 512

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        return x1, x2, x3, x4, x5


class UNetDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.up5 = self.make_up_layer(512, 512)
        self.conv5 = self.make_conv_layer(1024, 512)

        self.up6 = self.make_up_layer(512, 256)
        self.conv6 = self.make_conv_layer(512, 256)

        self.up7 = self.make_up_layer(256, 128)
        self.conv7 = self.make_conv_layer(256, 128)

        self.up8 = self.make_up_layer(128, 64)
        self.conv8 = self.make_conv_layer(128, 64)

    def forward(self, x5: torch.Tensor, x4: torch.Tensor, x3: torch.Tensor, x2: torch.Tensor, x1: torch.Tensor):
        d4 = self.conv5(torch.cat([self.up5(x5), x4], dim=1))
        d3 = self.conv6(torch.cat([self.up6(d4), x3], dim=1))
        d2 = self.conv7(torch.cat([self.up7(d3), x2], dim=1))
        d1 = self.conv8(torch.cat([self.up8(d2), x1], dim=1))
        return d1

    @staticmethod
    def make_up_layer(in_ch: int, out_ch: int) -> torch.nn.Module:
        return torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    @staticmethod
    def make_conv_layer(in_ch: int, out_ch: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )


class UNetVGG16(Model):
    def __init__(
            self,
            n_classes: int,
            apply_softmax: bool = False,
            get_weights: bool = True,
            **kwargs
    ):
        super().__init__()

        self._name = "UNetVGG"
        self._config = {
            "n_classes": n_classes,
            "apply_softmax": apply_softmax,
        }

        self.encoder = VGG16Encoder(get_weights)
        self.decoder = UNetDecoder()
        self.head = torch.nn.Conv2d(64, n_classes, kernel_size=1)
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)
        features = self.decoder(x5, x4, x3, x2, x1)
        logits = self.head(features)
        return torch.nn.functional.softmax(logits, dim=1) if self.apply_softmax else logits

    def get_encoder_modules(self) -> list[torch.nn.Module]:
        return [self.encoder]

    def get_decoder_modules(self) -> list[torch.nn.Module]:
        return [self.decoder]

    def get_encoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_encoder_modules() for p in m.parameters()]

    def get_decoder_parameters(self) -> list[torch.nn.parameter.Parameter]:
        return [p for m in self.get_decoder_modules() for p in m.parameters()]

    def change_n_classes(self, n_classes: int):
        self.head = torch.nn.Conv2d(64, n_classes, kernel_size=1)
        self._config["n_classes"] = n_classes


# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNetVGG16(len(dataset.label_id_to_name))
    model.eval()

    print(f"parameter count: {model.count_parameters() / 1000000:.2f}M")
    print(f"in  shape: {next(iter(dataloader))[0].shape}")
    print(f"out shape: {model(next(iter(dataloader))[0]).shape}")
