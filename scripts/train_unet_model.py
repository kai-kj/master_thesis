# %%

from pathlib import Path
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision.transforms.v2 as tv_transforms
import sklearn
import json
import shutil

from scripts.util.train import run_epoch
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI
from scripts.util.seed import seed

# %%

models_path = Path("models")
assert models_path.is_dir()

base_model_path = None
# base_model_path = models_path / "20260402_172703"
assert models_path is None or models_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

device = torch.device("cuda")
batch_size = 6

seed(42)

# %%

xy_transform = tv_transforms.Compose([
    tv_transforms.RandomHorizontalFlip(),
    tv_transforms.RandomVerticalFlip(),
    tv_transforms.RandomRotation([0, 90]),
])

x_transform = tv_transforms.Compose([
    tv_transforms.Normalize(dataset_stats["x_mean"], dataset_stats["x_std"]),
])

dataset_all = DatasetTreeAI("ID", "train", 0)
train_idx, valid_idx = sklearn.model_selection.train_test_split(range(len(dataset_all)), test_size=0.2)

dataset_train = DatasetTreeAI("ID", "train", 0, train_idx, xy_transform=xy_transform, x_transform=x_transform)
dataset_valid = DatasetTreeAI("ID", "train", 0, valid_idx, xy_transform=xy_transform, x_transform=x_transform)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True
)

dataloader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True
)


# %%

def preview_lr(_scheduler: torch.optim.lr_scheduler.LRScheduler, _num_epochs: int):
    lrs = []

    for _ in range(_num_epochs):
        lrs.append(_scheduler.get_last_lr()[0])
        _scheduler.step()

    fig, ax = plt.subplots()
    ax.scatter(range(_num_epochs), lrs, marker=".")
    ax.set_ylim(0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("lr")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.show()


class_counts = np.array(list(dataset_stats["y_counts"].values())) + 1  # avoid zero division
class_freq = class_counts / np.sum(class_counts)

weights = np.log(1 / class_freq)
weights = weights / weights.sum() * weights.shape[0]

model = UNet(len(dataset_stats["y_counts"])).to(device)

if base_model_path is not None:
    model.load_state_dict(torch.load(base_model_path / "model.pt"))

base_lr = 5e-4
num_train_epochs = 100
num_warmup_epochs = 0
num_total_epochs = num_warmup_epochs + num_train_epochs

loss_fn = torch.nn.CrossEntropyLoss(torch.Tensor(weights).to(device))
scaler = torch.amp.grad_scaler.GradScaler(device=device.type)
# scaler = None

optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_epochs, eta_min=base_lr * 0.1)

# scheduler = torch.optim.lr_scheduler.SequentialLR(
#     optimizer,
#     [
#         torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=base_lr * 0.1, total_iters=num_warmup_epochs),
#         torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_epochs, eta_min=base_lr * 0.1),
#     ],
#     [num_warmup_epochs],
# )

# preview_lr(scheduler, num_total_epochs)

# %%

model_path = models_path / time.strftime("%Y%m%d_%H%M%S")
model_path.mkdir(parents=True, exist_ok=False)

shutil.copy("scripts/train_unet_model.py", model_path / "train.py.txt")

history = []

start_time = time.time()

for epoch in range(num_total_epochs):
    print(f"epoch {epoch + 1}/{num_total_epochs}")

    curr_train_loss, curr_valid_loss, curr_lr = run_epoch(
        device,
        model,
        optimizer, scheduler, loss_fn,
        dataloader_train, dataloader_valid,
        scaler
    )

    torch.save(model.state_dict(), model_path / "model.pt")

    history.append((epoch, curr_lr, curr_train_loss, curr_valid_loss))

    print(f"  curr_lr: {history[-1][1]:.6f}\n  train loss: {history[-1][2]:.6f}\n  valid loss: {history[-1][3]:.6f}\n")

    if len(history) == 1 or history[-1][3] < np.min([h[3] for h in history[:-1]]):
        print("  best model, saving")
        torch.save(model.state_dict(), model_path / "model.pt")

    if (epoch + 1) % 5 == 0 or epoch == num_total_epochs - 1:
        fig, ax = plt.subplots()

        ax.plot([h[2] for h in history], label="train loss")
        ax.plot([h[3] for h in history], label="valid loss")

        ax.set_xlim(0, num_total_epochs)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(min(float(min(h[2], h[3])) for h in history), history[0][3])

        fig.tight_layout()
        fig.savefig(model_path / "loss.png")
        fig.show()

    current_time = time.time()
    time_per_epoch = (current_time - start_time) / (epoch + 1)
    remaining_time = time_per_epoch * (num_total_epochs - epoch - 1)

    print(f"  {time.strftime("%H:%M:%S", time.gmtime(time_per_epoch))} per epoch")
    print(f"  {time.strftime("%H:%M:%S", time.gmtime(remaining_time))} remaining")

end_time = time.time()

loss = pd.DataFrame(history, columns=["epoch", "lr", "train_loss", "valid_loss"])
loss.to_csv(model_path / "loss.csv", index=False)

duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

print(f"done, training time for {num_total_epochs} epochs: {duration}")
