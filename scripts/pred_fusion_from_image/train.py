# %%

import torch
from pathlib import Path
import time
import pandas as pd

from scripts.models.unet_res import UNetRes
from scripts.pred_fusion_from_image.model import UnetFusionPredictor
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations_strong, make_base_test_augmentations

# %%

base_mode_path = Path("models/unet_20260612_122705")
assert base_mode_path.is_dir()

device = "cuda"

# %%

base_model = UNetRes.load(base_mode_path / "model.json", base_mode_path / "model.pt")
model = UnetFusionPredictor(base_model, 16, freeze_encoder=True).to(device)

# %%

n_epochs = 25

optimizer = torch.optim.AdamW([
    {"params": model.head.parameters(), "lr": 1e-5},
])

loss_fn = torch.nn.MSELoss()

batch_size = 16
train_aug = make_base_train_augmentations_strong()
test_aug = make_base_test_augmentations()

dataset = DatasetTreeAI("ID", "train", 0, return_clim=True, return_soil=True)
dataset_train, dataset_valid = dataset.split(0.2, train_aug, test_aug, 42)
dataloader_train, dataloader_valid = dataset_train.loader(batch_size), dataset_valid.loader(batch_size)

# %%

train_losses = []
valid_losses = []

for epoch in range(n_epochs):
    model.train()

    loss_total = 0.0
    loss_count = 0

    for x, y_true, _ in dataloader_train:
        x, y_true = x.to(device), y_true.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        loss_total += loss.item() * x.size(0)
        loss_count += x.size(0)

    train_loss = loss_total / loss_count
    train_losses.append(train_loss)

    model.eval()

    loss_total = 0.0
    loss_count = 0

    with torch.no_grad():
        for x, y_true, _ in dataloader_valid:
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)

            loss = loss_fn(y_pred, y_true)

            loss_total += loss.item() * x.size(0)
            loss_count += x.size(0)

    valid_loss = loss_total / loss_count
    valid_losses.append(valid_loss)

    print(f"epoch {epoch + 1}/{n_epochs}: train loss: {train_loss:.4f} | valid loss: {valid_loss:.4f}")

model_path = Path("models") / f"predictor_{time.strftime("%Y%m%d_%H%M%S")}"
model_path.mkdir(parents=True, exist_ok=True)
model.save(model_path / "model.json", model_path / "model.pt")

history = pd.DataFrame({"train_loss": train_losses, "valid_loss": valid_losses})
history.to_csv(model_path / "history.csv")
