# %%

from pathlib import Path
import time
import torch
import json

from scripts.train.loss import FocalTverskyLoss, calc_class_weights
from scripts.train.train import run_loop
from scripts.models.unet_scratch import UNet
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations, make_base_test_augmentations

# %%

models_path = Path("models")
assert models_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

dataset_train, dataset_valid = DatasetTreeAI("ID", "train", 0).split(
    0.2,
    train_augmentations=make_base_train_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
    test_augmentations=make_base_test_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
    random_state=42,
)

batch_size = 6
dataloader_train = dataset_train.loader(batch_size)
dataloader_valid = dataset_valid.loader(batch_size)

# %%

device = torch.device("cuda")

base_lr = 5e-4
num_epochs = 200

model = UNet(len(dataset_stats["y_counts"]), norm="instance").to(device)
loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr * 0.01)

# %%

run_loop(
    device, model, optimizer, scheduler, loss_fn,
    num_epochs, dataloader_train, dataloader_valid,
    models_path / f"unet_ftv_{time.strftime("%Y%m%d_%H%M%S")}",
    len(dataset_stats["y_counts"])
)
