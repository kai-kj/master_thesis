# %%

from pathlib import Path
import time
import torch
import torchvision.transforms.v2 as tv_transforms
import json
import numpy as np

from scripts.train.train import run_loop
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI

# %%

models_path = Path("models")
assert models_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

dataset_all = DatasetTreeAI(
    "ID", "train", 0,
    xy_transform=tv_transforms.Compose([
        tv_transforms.RandomHorizontalFlip(),
        tv_transforms.RandomVerticalFlip(),
        tv_transforms.RandomRotation([0, 90]),
    ]),
    x_transform=tv_transforms.Compose([
        tv_transforms.Normalize(dataset_stats["x_mean"], dataset_stats["x_std"]),
    ]),
)

dataset_train, dataset_valid = dataset_all.split(0.2, random_state=42)

batch_size = 3
dataloader_train = dataset_train.loader(batch_size)
dataloader_valid = dataset_valid.loader(batch_size)

# %%

base_lr = 1e-3
num_epochs = 100

class_counts = np.array(list(dataset_stats["y_counts"].values())) + 1  # avoid zero division
class_freq = class_counts / np.sum(class_counts)

weights = np.log(1 / class_freq)
weights = weights / weights.sum() * weights.shape[0]

device = torch.device("cuda")
model = UNet(len(dataset_stats["y_counts"]), norm="group").to(device)
loss_fn = torch.nn.CrossEntropyLoss(torch.Tensor(weights).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr * 0.01)

# %%

run_loop(
    device, model, optimizer, scheduler, loss_fn,
    num_epochs, dataloader_train, dataloader_valid,
    models_path / time.strftime("%Y%m%d_%H%M%S")
)
