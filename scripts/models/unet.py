# %%

import torch


# %%

class UNet(torch.nn.Module):
    def __init__(self, n_classes, apply_softmax=False):
        super(UNet, self).__init__()

        # in -> 1024x1024x3
        self.conv11 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)  # -> 1024x1024x64
        self.conv12 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)  # -> 1024x1024x64
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 512x512x64

        self.conv21 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)  # -> 512x512x128
        self.conv22 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)  # -> 512x512x128
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 256x256x128

        self.conv31 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)  # -> 256x256x256
        self.conv32 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)  # -> 256x256x256
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 128x128x256

        self.conv41 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)  # -> 128x128x512
        self.conv42 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)  # -> 128x128x512
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # -> 64x64x512

        self.conv51 = torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # -> 64x64x1024
        self.conv52 = torch.nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # -> 64x64x1024
        self.up5 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # -> 128x128x512

        # (cat -> 128x128x1024)
        self.conv61 = torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1)  # -> 128x128x512
        self.conv62 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)  # -> 128x128x512
        self.up6 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # -> 256x256x256

        # (cat -> 256x256x512)
        self.conv71 = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)  # -> 256x256x256
        self.conv72 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)  # -> 256x256x256
        self.up7 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # -> 512x512x128

        # (cat -> 512x512x256)
        self.conv81 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)  # -> 512x512x128
        self.conv82 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)  # -> 512x512x128
        self.up8 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # -> 1024x1024x64

        # (cat -> 1024x1024x128)
        self.conv91 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)  # -> 1024x1024x64
        self.conv92 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)  # -> 1024x1024x64
        self.conv93 = torch.nn.ConvTranspose2d(64, n_classes, kernel_size=1)  # -> 1024x1024xn_classes

        self.apply_softmax = apply_softmax

    def forward(self, x11):
        x12 = torch.nn.functional.relu(self.conv11(x11))
        x13 = torch.nn.functional.relu(self.conv12(x12))
        x21 = self.pool1(x13)

        x22 = torch.nn.functional.relu(self.conv21(x21))
        x23 = torch.nn.functional.relu(self.conv22(x22))
        x31 = self.pool2(x23)

        x32 = torch.nn.functional.relu(self.conv31(x31))
        x33 = torch.nn.functional.relu(self.conv32(x32))
        x41 = self.pool3(x33)

        x42 = torch.nn.functional.relu(self.conv41(x41))
        x43 = torch.nn.functional.relu(self.conv42(x42))
        x51 = self.pool4(x43)

        x52 = torch.nn.functional.relu(self.conv51(x51))
        x53 = torch.nn.functional.relu(self.conv52(x52))
        x61 = torch.cat((self.up5(x53), x43), dim=1)

        x62 = torch.nn.functional.relu(self.conv61(x61))
        x63 = torch.nn.functional.relu(self.conv62(x62))
        x71 = torch.cat((self.up6(x63), x33), dim=1)

        x72 = torch.nn.functional.relu(self.conv71(x71))
        x73 = torch.nn.functional.relu(self.conv72(x72))
        x81 = torch.cat((self.up7(x73), x23), dim=1)

        x82 = torch.nn.functional.relu(self.conv81(x81))
        x83 = torch.nn.functional.relu(self.conv82(x82))
        x91 = torch.cat((self.up8(x83), x13), dim=1)

        x92 = torch.nn.functional.relu(self.conv91(x91))
        x93 = torch.nn.functional.relu(self.conv92(x92))
        x101 = self.conv93(x93)

        return torch.nn.functional.softmax(x101, dim=1) if self.apply_softmax else x101


# %%

if __name__ == "__main__":
    from scripts.dataset import DatasetTreeAI

    dataset = DatasetTreeAI("ID", "train", 0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(len(dataset.label_id_to_name))
    model.eval()

    torch.onnx.export(
        model,
        (next(iter(dataloader))[0],),
        "models/untrained_onnx/unet.onnx",
        input_names=["input"],
        output_names=["output"],
        external_data=False,
        dynamo=True,
    )
