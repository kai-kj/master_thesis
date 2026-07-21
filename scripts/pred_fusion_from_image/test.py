# %%

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from scripts.pred_fusion_from_image.model import UnetFusionPredictor
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations_strong, make_base_test_augmentations

# %%

model_path = Path("models/predictor_20260709_135554")
assert model_path.is_dir()

device = "cuda"

model = UnetFusionPredictor.load(model_path / "model.json", model_path / "model.pt").to(device)
history = pd.read_csv(model_path / "history.csv")

# %%

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(history["train_loss"], label="train loss")
ax.plot(history["valid_loss"], label="validation loss")

ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig(model_path / "loss.png")
fig.show()

# %%

batch_size = 16
train_aug = make_base_train_augmentations_strong()
test_aug = make_base_test_augmentations()

dataset = DatasetTreeAI(
    "ID", "test", 0,
    return_clim=True, return_soil=True,
    augmentations=make_base_test_augmentations()
)

dataloader = dataset.loader(batch_size)

# %%

model.eval()

ys_pred = []
ys_true = []

with torch.no_grad():
    for x, y_true, _ in dataloader:
        ys_pred.append(model(x.to(device)).detach().cpu().numpy())
        ys_true.append(y_true.detach().cpu().numpy())

ys_pred = np.concatenate(ys_pred, axis=0)
ys_true = np.concatenate(ys_true, axis=0)

# %%

r2_each: np.ndarray = r2_score(ys_true, ys_pred, multioutput="raw_values")  # type: ignore
mae_each: np.ndarray = mean_absolute_error(ys_true, ys_pred, multioutput="raw_values")  # type: ignore
rmse_each: np.ndarray = root_mean_squared_error(ys_true, ys_pred, multioutput="raw_values")  # type: ignore

r2_mean = r2_each.mean().item()
mae_mean = mae_each.mean().item()
rmse_mean = rmse_each.mean().item()

print(f"R^2: {r2_mean:.4f}")
print(f"MAE: {mae_mean:.4f}")
print(f"RMSE: {rmse_mean:.4f}")

(model_path / "results.json").write_text(json.dumps(
    {
        "r2": {
            "each": {k: v for k, v in zip(dataset.get_fusion_columns(), r2_each.tolist())},
            "average": r2_mean
        },
        "mae": {
            "each": {k: v for k, v in zip(dataset.get_fusion_columns(), mae_each.tolist())},
            "average": mae_mean
        },
        "rmse": {
            "each": {k: v for k, v in zip(dataset.get_fusion_columns(), rmse_each.tolist())},
            "average": rmse_mean
        },
    },
    indent=4,
))

# %%


fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(np.arange(ys_true.shape[1]), r2_each)

ax.set_ylabel("$r^2$")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(np.arange(ys_true.shape[1]), dataset.get_fusion_columns(), rotation=90)

fig.tight_layout()
fig.savefig(model_path / "r2_each.png")
fig.show()
