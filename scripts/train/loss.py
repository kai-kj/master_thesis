import torch
from typing import Literal


class TverskyLoss(torch.nn.Module):
    def __init__(
            self, alpha: float = 0.5, beta: float = 0.5,
            class_weights: torch.Tensor | None = None,
            reduction: Literal["mean", "sum", "none"] = "mean",
            smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.nn.functional.softmax(logits, dim=1)

        target = torch.nn.functional.one_hot(target, logits.shape[1])
        target = target.permute(0, 3, 1, 2).float()

        tp = (probs * target).sum([2, 3])
        fp = (probs * (1 - target)).sum([2, 3])
        fn = ((1 - probs) * target).sum([2, 3])

        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.class_weights is not None:
            tv = tv * self.class_weights

        match self.reduction:
            case "mean":
                return (1.0 - tv).mean()
            case "sum":
                return (1.0 - tv).sum()
            case "none":
                return 1.0 - tv


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, reduction="none", smooth=smooth)
        self.gamma = gamma

    def forward(self, logits, target):
        return torch.pow(self.tversky(logits, target), self.gamma).mean()


class PerImageHallucinationLoss(torch.nn.Module):
    def __init__(self, base_loss: torch.nn.Module, lambda_absent: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_absent = lambda_absent

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor):
        gt = torch.stack([(y_true == c).any((1, 2)) for c in range(logits.shape[1])], dim=1).float()
        pred = logits.softmax(1).amax((2, 3))
        return self.base_loss(logits, y_true) + self.lambda_absent * (pred * (1 - gt)).mean()


def calc_class_weights(class_counts: torch.Tensor, log_smooth: bool = True) -> torch.Tensor:
    zero_mask = class_counts == 0

    class_freq = class_counts / class_counts.sum()
    weights = 1 / class_freq

    if log_smooth:
        weights = torch.log(weights)

    weights[zero_mask] = 1

    return weights / weights.sum() * weights.shape[0]  # normalize around 1.0
