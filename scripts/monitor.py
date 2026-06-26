# %%

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np


# %%

def make_history_plot(model_path: Path, ax: Axes) -> tuple[float, float]:
    history: pd.DataFrame = None  # type: ignore

    if (model_path / "loss.csv").exists():
        history = pd.read_csv(model_path / "loss.csv")
    elif (model_path / "history.csv").exists():
        history = pd.read_csv(model_path / "history.csv")
    else:
        raise RuntimeError(f"loss.csv or history.csv not found")

    cm_columns = [c for c in history.columns if c.startswith("cm_")]
    cm = history[cm_columns].values.reshape(-1, int(np.sqrt(len(cm_columns))), int(np.sqrt(len(cm_columns))))

    tp = np.diagonal(cm, axis1=1, axis2=2)
    fp = cm.sum(axis=1) - tp
    fn = cm.sum(axis=2) - tp
    tn = cm.sum(axis=(1, 2))[:, None] - (tp + fp + fn)

    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn) != 0)

    support = tp + fn
    weights = np.divide(
        support, support.sum(axis=1, keepdims=True),
        out=np.zeros_like(support, dtype=float), where=support.sum(axis=1, keepdims=True) != 0
    )

    f1_weighted = (f1 * weights).sum(axis=1)
    iou_weighted = (iou * weights).sum(axis=1)

    # print(f"best f1:  {f1_weighted.max():.03f}")
    # print(f"best iou: {iou_weighted.max():.03f}")

    ax1 = ax
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    cc = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    epochs = np.arange(len(history["train_loss"])) + 1

    p1, = ax1.plot(epochs, history["lr"], color=next(cc), linestyle=":", label="learning rate")
    p2, = ax2.plot(epochs, history["train_loss"], color=next(cc), linestyle="--", label="train loss")
    p3, = ax2.plot(epochs, history["valid_loss"], color=next(cc), linestyle="--", label="valid loss")
    p4, = ax3.plot(epochs, f1_weighted, color=next(cc), label="f1 score")
    p5, = ax3.plot(epochs, iou_weighted, color=next(cc), label="iou score")

    min_loss, min_loss_e = float(history["valid_loss"].min()), float(history["valid_loss"].argmin()) + 1
    ax2.scatter(min_loss_e, min_loss, color="k", marker="^", zorder=10)
    ax2.text(min_loss_e, min_loss + 0.002 * min_loss, f"{min_loss:.02f}", color="k", zorder=10)

    min_f1, min_f1_e = float(f1_weighted.max()), float(f1_weighted.argmax()) + 1
    ax3.scatter(min_f1_e, min_f1, color="k", marker="^", zorder=10)
    ax3.text(min_f1_e, min_f1 + 0.02 * min_f1, f"{min_f1:.02f}", color="k", zorder=10)

    min_iou, min_iou_e = float(iou_weighted.max()), float(iou_weighted.argmax()) + 1
    ax3.scatter(min_iou_e, min_iou, color="k", marker="^", zorder=10)
    ax3.text(min_iou_e, min_iou + 0.02 * min_iou, f"{min_iou:.02f}", color="k", zorder=10)

    ax1.legend(handles=[p2, p3, p4, p5, p1], loc="upper right", ncols=3, frameon=False)

    ax1.set_xlabel("epoch")
    ax1.set_ylim(0.0, ax1.get_ylim()[1] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.125)
    ax1.set_ylabel("learning rate")
    ax1.spines[["top"]].set_visible(False)

    ax2.set_ylabel("loss")
    ax2.spines[["top"]].set_visible(False)

    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("score")
    ax3.spines[["top"]].set_visible(False)
    ax3.spines.right.set_position(("axes", 1.15))

    return f1_weighted[history["valid_loss"].argmin()], iou_weighted[history["valid_loss"].argmin()]


# %%

model_path = Path("models/unet_film_fusion_clim_soil_20260623_152617")
assert model_path.is_dir()

fig, ax = plt.subplots(figsize=(8, 6))

make_history_plot(model_path, ax)

fig.tight_layout()
fig.savefig(model_path / "loss.png")
fig.show()

# %%

# base_model_path = Path("models/swin_20260612_203922")
# assert base_model_path.is_dir()
#
# dists = ["china", "czechia", "iran", "mexico", "russia"]
# ks = [1, 5, 25, 125]
#
# f1_scores = {dist: [] for dist in dists}
# iou_scores = {dist: [] for dist in dists}
#
# fig, ax = plt.subplots(len(ks), len(dists), figsize=(8 * len(dists), 6 * len(ks)))
#
# for i, dist in enumerate(dists):
#     for j, k in enumerate(ks):
#         ft_model_path_pattern = base_model_path.name + "_" + dist + "_" + str(k) + "_shot_*"
#         ft_model_path = sorted(list(base_model_path.parent.glob(ft_model_path_pattern)))
#
#         if len(ft_model_path) == 0:
#             ax[j, i].axis("off")
#             continue
#
#         ft_model_path = ft_model_path[0]
#
#         res = make_history_plot(ft_model_path, ax[j, i])
#         ax[j, i].set_title(f"{dist} - {k} shot")
#
#         f1_scores[dist].append(res[0])
#         iou_scores[dist].append(res[1])
#
# fig.tight_layout()
# fig.savefig(base_model_path / f"loss_all.png")
# fig.show()
#
# # %%
#
# fig, ax = plt.subplots()
#
# for dist in dists:
#     ax.plot(f1_scores[dist], linestyle="-", marker="x", label=dist)
#
# ax.legend(loc="lower right", frameon=False)
# ax.set_xlabel("k")
# ax.set_ylabel("F1 score")
# ax.set_xticks(range(len(ks)), [str(k) for k in ks])
# ax.set_ylim(0.5, 1)
#
# ax.spines[["right", "top"]].set_visible(False)
#
# fig.tight_layout()
# fig.show()
