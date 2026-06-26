import torch


class FusionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._config = {}

    def forward(self, image_features: torch.Tensor, fusion_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FusionConcat(FusionModule):
    def __init__(self, n_image_encoder_features: int, n_fusion_encoder_features: int):
        super().__init__()

        self.projector = torch.nn.Linear(n_fusion_encoder_features, n_image_encoder_features)
        self.fuser = torch.nn.Sequential(
            torch.nn.Conv2d(n_image_encoder_features * 2, n_image_encoder_features, kernel_size=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, image_features: torch.Tensor, fusion_features: torch.Tensor) -> torch.Tensor:
        fusion_features = self.projector(fusion_features)

        fusion_features = fusion_features[:, :, None, None]
        fusion_features = fusion_features.expand(-1, -1, image_features.shape[2], image_features.shape[3])

        return self.fuser(torch.cat([image_features, fusion_features], dim=1))


class FusionFILM(FusionModule):
    def __init__(self, n_image_encoder_features: int, n_fusion_encoder_features: int):
        super().__init__()
        self.film = torch.nn.Linear(n_fusion_encoder_features, 2 * n_image_encoder_features)

    def forward(self, image_features: torch.Tensor, fusion_features: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(fusion_features).chunk(2, dim=1)
        gamma, beta = gamma[:, :, None, None], beta[:, :, None, None]

        return gamma * image_features + beta


class FusionLogit(FusionModule):
    def __init__(self, n_image_encoder_features: int, n_fusion_encoder_features: int):
        super().__init__()
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(n_fusion_encoder_features, n_image_encoder_features),
            torch.nn.ReLU()
        )

    def forward(self, image_features: torch.Tensor, fusion_features: torch.Tensor) -> torch.Tensor:
        return image_features + self.projector(fusion_features)[:, :, None, None]
