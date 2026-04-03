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

from scripts.util.train import run_epoch
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI
from scripts.util.seed import seed

# %%

models_path = Path("models")
assert models_path.is_dir()

# base_model_path = None
base_models_path = models_path / "20260402_172703"
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

class_counts = np.array(list(dataset_stats["y_counts"].values())) + 1  # avoid zero division
class_freq = class_counts / np.sum(class_counts)

weights = np.log(1 / class_freq)
weights = weights / weights.sum() * weights.shape[0]

model = UNet(len(dataset_stats["y_counts"])).to(device)

if base_models_path is not None:
    model.load_state_dict(torch.load(base_models_path / "model.pt"))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(torch.Tensor(weights).to(device))
scaler = torch.amp.grad_scaler.GradScaler(device=device.type)

# %%

model_path = models_path / time.strftime("%Y%m%d_%H%M%S")
model_path.mkdir(parents=True, exist_ok=False)

(model_path / "class.txt").write_text(str(type(model)))

num_epochs = 25
train_loss = []
valid_loss = []

start_time = time.time()

for epoch in range(num_epochs):
    print(f"epoch {epoch + 1}/{num_epochs}")

    curr_train_loss, curr_valid_loss = run_epoch(
        device,
        model,
        optimizer, loss_fn,
        dataloader_train, dataloader_valid,
        scaler
    )

    torch.save(model.state_dict(), model_path / "model.pt")

    train_loss.append(curr_train_loss)
    valid_loss.append(curr_valid_loss)

    print(f"  train loss: {train_loss[-1]:.4f}\n  valid loss: {valid_loss[-1]:.4f}")

    if len(valid_loss) == 1 or valid_loss[-1] < np.min(valid_loss[:-1]):
        print("  best model, saving")
        torch.save(model.state_dict(), model_path / "model.pt")

    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        fig, ax = plt.subplots()

        ax.plot(train_loss, label="train loss")
        ax.plot(valid_loss, label="valid loss")

        ax.set_xlim(0, num_epochs)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(model_path / "loss.png")
        fig.show()

    current_time = time.time()
    time_per_epoch = (current_time - start_time) / (epoch + 1)
    remaining_time = time_per_epoch * (num_epochs - epoch - 1)

    print(f"  {time.strftime("%H:%M:%S", time.gmtime(time_per_epoch))} per epoch")
    print(f"  {time.strftime("%H:%M:%S", time.gmtime(remaining_time))} remaining")

end_time = time.time()

loss = pd.DataFrame(
    zip(list(i + 1 for i in range(len(train_loss))), train_loss, valid_loss),
    columns=["epoch", "train_loss", "valid_loss"]
)

loss.to_csv(model_path / "loss.csv", index=False)

duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

print(f"done, training time for {num_epochs} epochs: {duration}")
