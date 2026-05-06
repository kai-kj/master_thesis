# %%

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%

model_paths = {
    "id": Path("models/unet_ftv_20260505_023132"),
    "china": Path("models/unet_ftv_20260505_023132_china_20260505_133841"),
    "czechia": Path("models/unet_ftv_20260505_023132_czechia_20260505_130023"),
    "iran": Path("models/unet_ftv_20260505_023132_iran_20260505_133957"),
    "mexico": Path("models/unet_ftv_20260505_023132_mexico_20260505_134553"),
    "russia": Path("models/unet_ftv_20260505_023132_russia_20260505_134926"),
    "south_africa": Path("models/unet_ftv_20260505_023132_south_africa_20260505_135708"),
}

model_names = {
    "id": "base model",
    "china": "China FT model",
    "czechia": "Czechia FT model",
    "iran": "Iran FT model",
    "mexico": "Mexico FT model",
    "russia": "Russia FT model",
    "south_africa": "South Africa FT model",
}

for model_path in model_paths.values():
    assert model_path.is_dir()

# %%

pre_ft_performances = []
post_ft_performances = []
post_ft_id_performances = []

for model_id, model_path in model_paths.items():
    if model_id != "south_africa":
        pre_ft_performance = pd.read_csv(model_paths["id"] / f"metrics_{model_id}.csv")
        pre_ft_performances.append((pre_ft_performance["f1"][2], pre_ft_performance["iou"][2]))
    else:
        pre_ft_performances.append((0, 0))

    if model_id != "id":
        post_ft_performance = pd.read_csv(model_path / f"metrics_{model_id}.csv")
        post_ft_performances.append((post_ft_performance["f1"][2], post_ft_performance["iou"][2]))
    else:
        post_ft_performances.append((0, 0))

    if model_id != "id" and model_id != "south_africa":
        post_ft_id_performance = pd.read_csv(model_path / f"metrics_id.csv")
        post_ft_id_performances.append((post_ft_id_performance["f1"][2], post_ft_id_performance["iou"][2]))
    else:
        post_ft_id_performances.append((0, 0))

summary = [
    (v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1])
    for v in zip(pre_ft_performances, post_ft_performances, post_ft_id_performances)
]

summary = pd.DataFrame(
    summary,
    columns=["pre_ft_f1", "pre_ft_iou", "post_ft_f1", "post_ft_iou", "post_ft_id_f1", "post_ft_id_iou"],
)

summary.insert(0, "dist", model_paths.keys())  # type: ignore

# %%

for metric_id, metric_name in [("f1", "weighted F1 score"), ("iou", "weighted IoU")]:
    fig, ax = plt.subplots(figsize=(10, 6))

    xs_a = np.arange(len(summary["dist"])) * 4 + 0
    xs_b = np.arange(len(summary["dist"])) * 4 + 1
    xs_c = np.arange(len(summary["dist"])) * 4 + 2

    ax.vlines(xs_a, 0, 1.1, zorder=1, color="lightgrey")
    ax.vlines(xs_b, 0, 1.1, zorder=1, color="lightgrey")
    ax.vlines(xs_c, 0, 1.1, zorder=1, color="lightgrey")

    ax.bar(xs_a, summary[f"pre_ft_{metric_id}"], label="before fine-tuning, on target set")
    ax.bar(xs_b, summary[f"post_ft_{metric_id}"], label="after fine-tuning, on target set")
    ax.bar(xs_c, summary[f"post_ft_id_{metric_id}"], label="after fine-tuning, on ID set")

    for i, v in enumerate(summary[f"pre_ft_{metric_id}"]):
        if v != 0: ax.text(xs_a[i] - 0.5, v + 0.01, f"{v:.02f}")

    for i, v in enumerate(summary[f"post_ft_{metric_id}"]):
        if v != 0: ax.text(xs_b[i] - 0.5, v + 0.01, f"{v:.02f}")

    for i, v in enumerate(summary[f"post_ft_id_{metric_id}"]):
        if v != 0: ax.text(xs_c[i] - 0.5, v + 0.01, f"{v:.02f}")

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric_name)
    ax.set_xticks(np.arange(len(summary["dist"])) * 4 + 1, labels=[model_names[k] for k in model_paths], rotation=20)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(f"figures/tune_summary_{metric_id}.png")
    fig.show()
