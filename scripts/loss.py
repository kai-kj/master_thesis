import torch


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, target):
        num_classes = logits.shape[1]

        probs = torch.nn.functional.softmax(logits, dim=1)

        dims = (0, 2, 3)

        tp = (probs * target).sum(dims)
        fp = (probs * (1 - target)).sum(dims)
        fn = ((1 - probs) * target).sum(dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth, reduction="none")
        self.gamma = gamma

    def forward(self, logits, target):
        tversky_loss = self.tversky(logits, target)
        focal_loss = torch.pow(tversky_loss, self.gamma)
        return focal_loss.mean()
