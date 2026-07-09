# %%

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

# %%

base_model_paths = [
    ("UNet", Path("models/unet_20260612_122705")),
    ("DeepLabV3+", Path("models/deeplab_20260612_135725")),
    ("TransUNet", Path("models/transunet_20260612_170208")),
    ("SwinFPN", Path("models/swin_20260612_203922")),
    ("Segformer", Path("models/segformer_20260612_145904")),

]

dist_names = ["austria", "china", "czechia", "iran", "mexico", "russia", "south_africa"]

ks = [1, 5, 25, 125]

# %%

metric = "iou"
agg = "weighted"

results = []

for base_mode_name, base_model_path in base_model_paths:
    result = pd.read_csv(base_model_path / f"metrics_id.csv").set_index("metric")
    results.append([base_mode_name, "id", 0, result.loc[agg, metric]])

    for dist in dist_names:
        result = pd.read_csv(base_model_path / f"metrics_{dist}.csv").set_index("metric")
        results.append([base_mode_name, dist, 0, result.loc[agg, metric]])

    for dist in dist_names:
        for k in ks:
            model_name = f"{base_model_path.name}_{dist}_{k}_shot"
            model_path = list(base_model_path.parent.glob(f"{model_name}_*"))

            if len(model_path) > 0:
                result = pd.read_csv(model_path[0] / f"metrics_{dist}.csv").set_index("metric")
                results.append([base_mode_name, dist, int(k), result.loc[agg, metric]])

results = pd.DataFrame(results, columns=["model", "dist", "k", "score"])

# %% base performance

fig, ax = plt.subplots()

ax.bar(results.loc[results["dist"] == "id", "model"], results.loc[results["dist"] == "id", "score"])

for i, (_, result) in enumerate(results.loc[results["dist"] == "id"].iterrows()):
    ax.text(i, result["score"] + 0.025, f"{result["score"]:.2f}", horizontalalignment="center")

# ax.set_ylim(0.75, 0.9)
ax.set_ylabel(f"{agg} {metric} score")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.show()

# %% tuning

fig, ax = plt.subplots(figsize=(10, 5))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for i, k in enumerate([0] + ks):
    ax.text(i * (len(base_model_paths) + 1) + 2, 1.0, f"k = {k}", horizontalalignment='center')

    for j, (base_mode_name, base_model_path) in enumerate(base_model_paths):
        result = results.loc[(results["model"] == base_mode_name) & (results["k"] == k), "score"]

        ax.boxplot(
            result, positions=[i * (len(base_model_paths) + 1) + j],
            widths=[0.75], showmeans=True,
            medianprops={"color": "black"},
            meanprops={"marker": "x", "markerfacecolor": "r", "markeredgecolor": "r"},
            flierprops={"marker": ".", "markerfacecolor": "k", "markeredgecolor": "k"},

        )

# ax.legend(loc="lower right", frameon=False)
# ax.set_xlabel("k")

ax.set_xticks(
    [i * (len(base_model_paths) + 1) + j for i in range(len(base_model_paths)) for j in range(len(ks) + 1)],
    [m[0] for m in base_model_paths] * (len(ks) + 1),
    rotation=90,
)

ax.set_ylabel(f"{agg} {metric} score")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.show()

# %%

# metric = "f1"
# agg = "weighted"
#
# fig, ax = plt.subplots(figsize=(6, 6))
#
# scores = []
#
# for _, base_model_path in base_model_paths:
#     base_model_metrics_path = base_model_path / "metrics_id.csv"
#     assert base_model_metrics_path.is_file()
#
#     df = pd.read_csv(base_model_metrics_path).set_index("metric")
#     scores.append(df.loc[agg, metric])
#
# ax.bar(range(len(scores)), scores)
#
# fig.tight_layout()
# fig.show()

#
# model_names = {
#     "id": "core model",
#     "china": "china model",
#     "czechia": "czechia model",
#     "iran": "iran model",
#     "mexico": "mexico model",
#     "russia": "russia model",
#     "south_africa": "south_africa model",
# }
#
# data_names = {
#     "id": "core data",
#     "china": "china data",
#     "czechia": "czechia data",
#     "iran": "iran data",
#     "mexico": "mexico data",
#     "russia": "russia data",
#     "south_africa": "south_africa data",
# }
#
# for model_path in model_paths.values():
#     assert model_path.is_dir()

# # %%
#
# metric = "f1"
# agg = "weighted"
#
# metrics = dict()
#
# columns = list(model_paths.keys())
#
# for model_id, model_path in model_paths.items():
#     model_metrics = dict()
#
#     for data_id in model_paths.keys():
#         metrics_path = model_path / f"metrics_{data_id}.csv"
#
#         if not metrics_path.is_file():
#             continue
#
#         df = pd.read_csv(metrics_path).set_index("metric")
#         model_metrics[data_names[data_id]] = df.loc[agg, metric]
#
#     metrics[model_names[model_id]] = model_metrics
#
# metrics = pd.DataFrame(metrics).transpose()
#
# # %%
#
# xs = []
# ys = []
#
# for data_name in metrics.keys():
#     for model_name in metrics.index:
#         value: np.float64 = metrics.loc[model_name, data_name]  # type: ignore
#
#         if not np.isnan(value):
#             xs.append(f"{model_name} on {data_name}")
#             ys.append(value)
#
#     xs.append("")
#     ys.append(0.0)
#
# xs = xs[:-1]
# ys = ys[:-1]
#
# fig, ax = plt.subplots(figsize=(8, 5))
#
# for model_name in model_names.values():
#     ax.bar(
#         [i for i in range(len(xs)) if xs[i].startswith(model_name)],
#         [ys[i] for i in range(len(xs)) if xs[i].startswith(model_name)],
#         align="center", label=model_name, hatch=".." if model_name == "core model" else "",
#     )
#
#     for i in [i for i in range(len(xs)) if xs[i].startswith(model_name)]:
#         ax.text(i - 0.45, (int(ys[i] * 100) + 1) / 100 + 0.01, f"{ys[i]:.02f}", size="small")
#
# ax.spines[["top", "right"]].set_visible(False)
# ax.set_ylim(0, 1.15)
# ax.set_ylabel(f"{agg} {metric} score")
# ax.set_xticks([2.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22], data_names.values(), rotation=30)
# ax.legend(loc="upper right", ncols=4, frameon=False)
#
# fig.tight_layout()
# fig.savefig(f"figures/tune_summary_{out_name}_{agg}_{metric}.png")
# fig.show()
