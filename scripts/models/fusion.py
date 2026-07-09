import torch


class FusionFILM(torch.nn.Module):
    def __init__(self, n_image_encoder_features: int, n_fusion_encoder_features: int):
        super().__init__()
        self.film = torch.nn.Linear(n_fusion_encoder_features, 2 * n_image_encoder_features)

    def forward(self, image_features: torch.Tensor, fusion_features: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(fusion_features).chunk(2, dim=1)
        gamma, beta = gamma[:, :, None, None], beta[:, :, None, None]

        return gamma * image_features + beta


class FusionAttention(torch.nn.Module):
    def __init__(
            self,
            n_image_features: int,
            n_fusion_features: int,
            n_heads: int = 8,
            n_fusion_tokens: int = 4,
    ):
        super().__init__()

        assert n_image_features % n_heads == 0

        self.n_image_features = n_image_features
        self.n_fusion_tokens = n_fusion_tokens

        self.fusion_projector = torch.nn.Linear(
            n_fusion_features,
            n_fusion_tokens * n_image_features,
        )

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=n_image_features,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm = torch.nn.LayerNorm(
            n_image_features
        )

    def forward(
            self,
            image_features,
            fusion_features,
    ):
        image_tokens = image_features.flatten(2).transpose(1, 2)

        fusion_tokens = self.fusion_projector(fusion_features)
        fusion_tokens = fusion_tokens.reshape(image_features.shape[0], self.n_fusion_tokens, image_features.shape[1])

        attended, _ = self.attention(
            query=image_tokens,
            key=fusion_tokens,
            value=fusion_tokens,
            need_weights=False,
        )

        return self.norm(image_tokens + attended).transpose(1, 2).reshape(*image_features.shape)
