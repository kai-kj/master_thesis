# %%

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%

model_path = Path("models/unet_ftv_20260505_023132_czechia_20260505_130023")
assert model_path.is_dir()

# %%

history: pd.DataFrame = None  # type: ignore

if (model_path / "loss.csv").exists():
    history = pd.read_csv(model_path / "loss.csv")
elif (model_path / "history.csv").exists():
    history = pd.read_csv(model_path / "history.csv")
else:
    raise RuntimeError(f"loss.csv or history.csv not found")

# %%

cm_columns = [c for c in history.columns if c.startswith("cm_")]
cm = history[cm_columns].values.reshape(-1, 10, 10)

tp = np.diagonal(cm, axis1=1, axis2=2)
fp = cm.sum(axis=1) - tp
fn = cm.sum(axis=2) - tp
tn = cm.sum(axis=(1, 2))[:, None] - (tp + fp + fn)

precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn) != 0)

support = tp + fn
weights = np.divide(
    support, support.sum(axis=1, keepdims=True),
    out=np.zeros_like(support, dtype=float), where=support.sum(axis=1, keepdims=True) != 0
)

precision_weighted = (precision * weights).sum(axis=1)
recall_weighted = (recall * weights).sum(axis=1)
f1_weighted = (f1 * weights).sum(axis=1)
iou_weighted = (iou * weights).sum(axis=1)

print(f"best f1:  {f1_weighted.max():.03f}")
print(f"best iou: {iou_weighted.max():.03f}")

# %%

fig, ax1 = plt.subplots(1, 2, figsize=(12, 6))
ax2 = ax1[0].twinx()

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
epochs = np.arange(len(history["train_loss"])) + 1

p1, = ax1[0].plot(epochs, history["train_loss"], color=color_cycle[0], label="train loss")
p2, = ax1[0].plot(epochs, history["valid_loss"], color=color_cycle[1], label="valid loss")
p3, = ax2.plot(epochs, history["lr"], color=color_cycle[2], linestyle="--", label="learning rate")

ax1[0].set_xlabel("epoch")
ax1[0].set_ylabel("loss")
ax1[0].spines[["top"]].set_visible(False)
ax1[0].legend(handles=[p1, p2, p3], loc="upper right", frameon=False)

ax2_ylim_margin = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.125
ax2.set_ylim(max(0, ax2.get_ylim()[0] - ax2_ylim_margin), ax2.get_ylim()[1] + ax2_ylim_margin)
ax2.set_ylabel("learning rate")
ax2.spines[["top"]].set_visible(False)

ax1[1].plot(epochs, precision_weighted, label="precision (weighted)")
ax1[1].plot(epochs, recall_weighted, label="recall (weighted)")
ax1[1].plot(epochs, f1_weighted, label="f1 (weighted)")
ax1[1].plot(epochs, iou_weighted, label="iou (weighted)")

ax1[1].set_xlabel("epoch")
ax1[1].set_ylabel("score")
ax1[1].spines[["top", "right"]].set_visible(False)
ax1[1].legend(loc="upper left", frameon=False)
ax1[1].set_ylim(0, 1)

fig.tight_layout()
fig.savefig(model_path / "loss.png")
fig.show()
