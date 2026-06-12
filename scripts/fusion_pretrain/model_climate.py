# %%

import torch

from scripts.fusion_pretrain.dataset import DatasetClimateSoil


# %%

class ClimateEncoder(torch.nn.Module):
    def __init__(self, n_in_features: int = 9, n_out_features: int = 32):
        super(ClimateEncoder, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in_features, n_out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(n_out_features, n_out_features),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClimateModel(torch.nn.Module):
    def __init__(self, n_classes: int, n_in_features: int = 9, n_out_features: int = 32):
        super(ClimateModel, self).__init__()

        self.encoder = ClimateEncoder(n_in_features=n_in_features, n_out_features=n_out_features)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_out_features, 2 * n_out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * n_out_features, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# %%

if __name__ == "__main__":
    dataset = DatasetClimateSoil("train", 0, include_clim=True)

    model = ClimateModel(n_classes=len(dataset.labels), n_in_features=len(dataset.features))

    x = dataset[0][0]
    y_true = dataset[0][1]
    y_pred = model.forward(x)

    print(f"x shape: {x.shape}")
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")
