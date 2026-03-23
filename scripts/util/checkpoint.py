import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return checkpoint(self.model, x)
