# %%

from pathlib import Path
from typing import Sized, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# %%

encoder_paths = {
    "climate": [
        Path("models/mlp_clim_split_0_20260717_120131"),
        Path("models/mlp_clim_split_1_20260717_122107"),
        Path("models/mlp_clim_split_2_20260717_124047"),
        Path("models/mlp_clim_split_3_20260717_125956"),
        Path("models/mlp_clim_split_4_20260717_131823"),
    ],
    "soil": [
        Path("models/mlp_soil_split_0_20260717_120758"),
        Path("models/mlp_soil_split_1_20260717_122736"),
        Path("models/mlp_soil_split_2_20260717_124723"),
        Path("models/mlp_soil_split_3_20260717_130604"),
        Path("models/mlp_soil_split_4_20260717_132432"),
    ],
    "climate+soil": [
        Path("models/mlp_clim_soil_split_0_20260717_121436"),
        Path("models/mlp_clim_soil_split_1_20260717_123406"),
        Path("models/mlp_clim_soil_split_2_20260717_125340"),
        Path("models/mlp_clim_soil_split_3_20260717_131212"),
        Path("models/mlp_clim_soil_split_4_20260717_133041"),
    ],
}

core_model_paths = {
    "UNet": [
        Path("models/unet_20260612_122705")
    ],
    "DeepLabV3+": [
        Path("models/deeplab_20260612_135725")
    ],
    "Segformer": [
        Path("models/segformer_20260612_145904")
    ],
    "TransUNet": [
        Path("models/transunet_20260612_170208")
    ],
    "SwinFPN": [
        Path("models/swin_20260612_203922")
    ],
}

predictor_paths = {
    "frozen encoder": [Path("models/predictor_20260709_135554")],
    "trainable encoder": [Path("models/predictor_20260709_130046")],
}

dist_names = ["austria", "china", "czechia", "iran", "mexico", "russia", "south_africa"]

ks = [1, 5, 25, 125]


# %%

def pad(value: Any, padding: float = 0.1) -> float:
    return value + value * padding


def bp_cfg(n_bars: int | Sized, width: float) -> dict:
    if not isinstance(n_bars, int):
        n_bars = len(n_bars)

    return {
        "widths": [width] * n_bars,
        # "showmeans": True,
        "medianprops": {"color": "black"},
        # "meanprops": {"marker": "x", "markerfacecolor": "r", "markeredgecolor": "r"},
        "flierprops": {"marker": ".", "markerfacecolor": "k", "markeredgecolor": "k"},
    }


# %% clim-soil pretraining - main

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

maes = {model_name: [] for model_name in encoder_paths.keys()}
bcs = {model_name: [] for model_name in encoder_paths.keys()}

for model_name, model_paths in encoder_paths.items():
    for model_path in model_paths:
        all_results = json.loads((model_path / "results.json").read_text())
        maes[model_name].append(all_results["mae"]["average"])
        bcs[model_name].append(all_results["bc"]["average"])

ax[0].boxplot(maes.values(), tick_labels=maes.keys(), **bp_cfg(bcs.keys(), 0.25))

for i, (_, vs) in enumerate(maes.items()):
    labels = f"mean: {np.mean(vs):.03f}"
    ax[0].text(i + 1, pad(max(vs), 0.01), labels, horizontalalignment="center")

ax[0].spines[["top", "right"]].set_visible(False)
ax[0].set_ylim(pad(min(min(v) for v in maes.values()), -0.05), pad(max(max(v) for v in maes.values()), 0.05))
ax[0].set_ylabel("MAE")
ax[0].set_title("MAE")

ax[1].boxplot(bcs.values(), tick_labels=bcs.keys(), **bp_cfg(bcs.keys(), 0.25))

for i, (_, vs) in enumerate(bcs.items()):
    labels = f"mean: {np.mean(vs):.03f}"
    ax[1].text(i + 1, pad(max(vs), 0.01), labels, horizontalalignment="center")

ax[1].spines[["top", "right"]].set_visible(False)
ax[1].set_ylim(pad(min(min(v) for v in bcs.values()), -0.05), pad(max(max(v) for v in bcs.values()), 0.05))
ax[1].set_ylabel("Bray-Curtis dissimilarity")
ax[1].set_title("Bray-Curtis dissimilarity")

fig.tight_layout()
fig.savefig("figures/pretraining_mae_bc.png")
fig.show()

# %% clim-soil pretraining - appendix

maes = {model_name: [] for model_name in encoder_paths.keys()}
bcs = {model_name: [] for model_name in encoder_paths.keys()}
labels = []

for i, (model_name, model_paths) in enumerate(encoder_paths.items()):
    for model_path in model_paths:
        all_results = json.loads((model_path / "results.json").read_text())

        if not labels: labels = list(all_results["mae"]["each"].keys())
        maes[model_name].append(list(all_results["mae"]["each"].values()))
        bcs[model_name].append(list(all_results["bc"]["each"].values()))

maes_mean = pd.DataFrame({k: np.mean(v, axis=0) for k, v in maes.items()}, index=labels).T
bcs_mean = pd.DataFrame({k: np.mean(v, axis=0) for k, v in bcs.items()}, index=labels).T
maes_std = pd.DataFrame({k: np.std(v, axis=0, ddof=1) for k, v in maes.items()}, index=labels, ).T
bcs_std = pd.DataFrame({k: np.std(v, axis=0, ddof=1) for k, v in bcs.items()}, index=labels, ).T

maes_mean["mean"] = {k: np.mean(np.mean(v, axis=1)) for k, v in maes.items()}
maes_std["mean"] = {k: np.std(np.mean(v, axis=1), ddof=1) for k, v in maes.items()}
bcs_mean["mean"] = {k: np.mean(np.mean(v, axis=1)) for k, v in bcs.items()}
bcs_std["mean"] = {k: np.std(np.mean(v, axis=1), ddof=1) for k, v in bcs.items()}

maes_table = maes_mean.astype(str)
bcs_table = bcs_mean.astype(str)

for col in maes_table.columns:
    maes_table[col] = [rf"${m:.3f}\pm{s:.3f}$" for m, s in zip(maes_mean[col], maes_std[col])]

for col in bcs_table.columns:
    bcs_table[col] = [rf"${m:.2f}\pm{s:.2f}$" for m, s in zip(bcs_mean[col], bcs_std[col])]

maes_table_latex = maes_table \
    .transpose() \
    .to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n")

bcs_table_latex = bcs_table \
    .transpose() \
    .to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n")

Path(f"tables/encoder_mae.tex").write_text(maes_table_latex)
Path(f"tables/encoder_bc.tex").write_text(bcs_table_latex)

# %%

metric = "f1"
agg = "weighted"

all_results = []

for base_mode_name, base_model_paths in core_model_paths.items():
    for split, base_model_path in enumerate(base_model_paths):
        result = pd.read_csv(base_model_path / f"metrics_id.csv").set_index("metric")
        all_results.append([base_mode_name, "id", 0, split, result.loc[agg, metric]])

        for dist in dist_names:
            result = pd.read_csv(base_model_path / f"metrics_{dist}.csv").set_index("metric")
            all_results.append([base_mode_name, dist, 0, split, result.loc[agg, metric]])

        for dist in dist_names:
            for k in ks:
                model_name = f"{base_model_path.name}_{dist}_{k}_shot"
                model_path = list(base_model_path.parent.glob(f"{model_name}_*"))

                if len(model_path) > 0:
                    result = pd.read_csv(model_path[0] / f"metrics_{dist}.csv").set_index("metric")
                    all_results.append([base_mode_name, dist, int(k), split, result.loc[agg, metric]])

all_results = pd.DataFrame(all_results, columns=["model", "dist", "k", "split", "score"])

best_scores = all_results[["dist", "score"]].groupby("dist").max()
best_scores = best_scores.rename(columns={"score": "best_score"}).reset_index()  # type: ignore

all_results = pd.merge(all_results, best_scores, on="dist", how="left")
all_results["rel_score"] = all_results["score"] / all_results["best_score"]
all_results = all_results.drop(columns=["best_score"])

all_results = all_results[~((all_results["dist"] != "id") & (all_results["k"] == 0))]

# %% core performance - main

core_results = {k: g["score"].values for k, g in all_results.loc[all_results["dist"] == "id"].groupby("model")}
core_results = {k: core_results[k] for k in core_model_paths.keys()}

fig, ax = plt.subplots(figsize=(12, 4))

ax.boxplot(core_results.values(), tick_labels=core_results.keys(), **bp_cfg(core_results.keys(), 0.25))  # type: ignore

for i, (_, vs) in enumerate(core_results.items()):
    labels = f"mean: {np.mean(vs):.03f}"  # type: ignore
    ax.text(i + 1, pad(max(vs), 0.0025), labels, horizontalalignment="center")

# ax.set_ylim(0.75, 0.9)
ax.set_ylabel(f"{agg} {metric} score")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("figures/core_model_performance.png")
fig.show()

# %% core performance - appendix

core_model_perf_table = {}

for base_model_name, base_model_paths in core_model_paths.items():
    split_results: list[pd.DataFrame] = []

    for base_model_path in base_model_paths:
        split_results.append(pd.read_csv(base_model_path / "metrics_id.csv").set_index("metric"))

    means: pd.DataFrame = sum(split_results) / len(split_results)
    stds = pd.concat(split_results).groupby(level=0).std(ddof=1).fillna(0.0)

    formatted = means.astype(str)  # type: ignore

    for col in formatted.columns:
        formatted[col] = [rf"${m:.3f}\pm{s:.3f}$" for m, s in zip(means[col], stds[col])]  # type: ignore

    core_model_perf_table[base_model_name] = formatted

core_model_perf_table = pd.concat(core_model_perf_table, names=["model", "metric"])

core_model_perf_table_latex = core_model_perf_table \
    .to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\\cline{1-6}", "") \
    .replace("\n ", "\n")

Path(f"tables/core.tex").write_text(core_model_perf_table_latex)

# %% tuning - by model - main

fig, ax = plt.subplots(figsize=(12, 4))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

mean_by_k = {k: [] for k in ks}

for i, k in enumerate(ks):
    ax.text(i * (len(core_model_paths) + 1) + 2, 1.025, f"k = {k}", horizontalalignment="center")

    for j, (base_mode_name, base_model_path) in enumerate(core_model_paths.items()):
        result = all_results.loc[(all_results["model"] == base_mode_name) & (all_results["k"] == k), "rel_score"]
        pos = i * (len(core_model_paths) + 1) + j

        ax.boxplot(result, positions=[pos], **bp_cfg(1, 0.5))
        mean_by_k[k].append((pos, result.mean().item()))

for k, vs in mean_by_k.items():
    ax.plot([pos for pos, _ in vs], [best for _, best in vs], color="gray", linestyle=":")

    best = max(v[1] for v in vs)

    for pos, v in vs:
        # ax.scatter(pos, v, color="red" if v == best else "orange", marker="^", s=50, zorder=10)
        ax.scatter(pos, v, color="red", marker="^", s=50, zorder=10, label="mean")

x_tick_locations = [j * (len(core_model_paths) + 1) + i for j in range(len(ks)) for i in
                    range(len(core_model_paths))]
ax.set_xticks(x_tick_locations, [m[0] for m in core_model_paths.items()] * (len(ks)), rotation=45)
ax.vlines(x_tick_locations, 0, 1.0, linestyles=":", color="lightgray", zorder=0)

plt.setp(ax.get_xticklabels(), ha="right")

ax.set_ylim(pad(all_results["rel_score"].quantile(0.05), -0.04), pad(1.0, 0.04))
ax.set_ylabel(f"relative {agg} {metric} score")
ax.spines[["top", "right"]].set_visible(False)
# ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig("figures/tuning_by_model.png")
fig.show()

# %% tuning - by site - main

fig, ax = plt.subplots(figsize=(12, 4))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

mean_by_k = {k: [] for k in ks}

for i, k in enumerate(ks):
    ax.text(i * (len(dist_names) + 1) + 2, 1.025, f"k = {k}", horizontalalignment="center")

    for j, dist_name in enumerate(dist_names):
        result = all_results.loc[(all_results["dist"] == dist_name) & (all_results["k"] == k), "score"]

        if len(result) == 0:
            continue

        pos = i * (len(dist_names) + 1) + j

        ax.boxplot(result, positions=[pos], **bp_cfg(1, 0.5))
        mean_by_k[k].append((pos, result.mean().item()))

x_tick_locations = [j * (len(dist_names) + 1) + i for j in range(len(ks)) for i in range(len(dist_names))]
ax.set_xticks(x_tick_locations, [d for d in dist_names] * (len(ks)), rotation=45)
ax.vlines(x_tick_locations, 0, 1.0, linestyles=":", color="lightgray", zorder=0)

plt.setp(ax.get_xticklabels(), ha="right")

# ax.set_ylim(pad(all_results["rel_score"].quantile(0.05), -0.04), pad(1.0, 0.04))
ax.set_ylabel(f"{agg} {metric} score")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("figures/tuning_by_site.png")
fig.show()

# %% tuning - appendix

tuning_perf_table_raw = []

for base_mode_name, base_model_paths in core_model_paths.items():
    for split, base_model_path in enumerate(base_model_paths):
        for dist in dist_names:
            result = pd.read_csv(base_model_path / f"metrics_{dist}.csv")
            result[["split", "model", "dist", "k"]] = [split, base_mode_name, dist, 0]
            tuning_perf_table_raw.append(result)

        for dist in dist_names:
            for k in ks:
                model_name = f"{base_model_path.name}_{dist}_{k}_shot"
                model_path = list(base_model_path.parent.glob(f"{model_name}_*"))

                if len(model_path) > 0:
                    result = pd.read_csv(model_path[0] / f"metrics_{dist}.csv")
                    result[["split", "model", "dist", "k"]] = [split, base_mode_name, dist, k]
                    tuning_perf_table_raw.append(result)

tuning_perf_table_raw = pd.concat(tuning_perf_table_raw).reset_index(drop=True)
tuning_perf_table_raw = tuning_perf_table_raw[tuning_perf_table_raw["metric"] == "weighted"]
tuning_perf_table_raw = tuning_perf_table_raw.drop(columns=["metric"])

tuning_perf_table_mean = tuning_perf_table_raw \
    .groupby(["dist", "model", "k"])[["precision", "recall", "f1", "iou"]] \
    .mean()

tuning_perf_table_std = tuning_perf_table_raw \
    .groupby(["dist", "model", "k"])[["precision", "recall", "f1", "iou"]] \
    .std(ddof=1) \
    .fillna(0.0)

tuning_perf_table = tuning_perf_table_mean.astype(str)  # type: ignore

for col in tuning_perf_table.columns:
    tuning_perf_table[col] = [
        rf"${m:.3f}\pm{s:.3f}$" for m, s in
        zip(tuning_perf_table_mean[col], tuning_perf_table_std[col])  # type: ignore
    ]

for dist in dist_names:
    tuning_perf_table_latex = tuning_perf_table.loc[dist].to_latex(escape=False) \
        .replace("\n\\toprule", "") \
        .replace("\n\\midrule", "\n\\hline") \
        .replace("\n\\bottomrule", "") \
        .replace("\\cline{1-6}", "") \
        .replace("\n ", "\n")

    Path(f"tables/{dist}_tune.tex").write_text(tuning_perf_table_latex)

# %% image -> clim+soil - main

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

mae_max = 0
r2_min = 1

for i, (pred_name, pred_paths) in enumerate(predictor_paths.items()):
    maes = []
    r2s = []

    for pred_path in pred_paths:
        result = json.loads((pred_path / "results.json").read_text())

        maes.append(result["mae"]["average"])
        mae_max = max(mae_max, result["mae"]["average"])

        r2s.append(result["r2"]["average"])
        r2_min = min(r2_min, result["r2"]["average"])

    ax[0].boxplot(maes, positions=[i], **bp_cfg(1, 0.5))
    ax[1].boxplot(r2s, positions=[i], **bp_cfg(1, 0.5))

ax[0].spines[["top", "right"]].set_visible(False)
ax[0].set_ylim(0, pad(mae_max, 0.25))
ax[0].set_ylabel("MAE")
ax[0].set_xticks(range(len(predictor_paths)), predictor_paths.keys())

ax[1].spines[["top", "right"]].set_visible(False)
ax[1].set_ylim(pad(r2_min, -0.05), 1)
ax[1].set_ylabel("$R^2$")
ax[1].set_xticks(range(len(predictor_paths)), predictor_paths.keys())

fig.tight_layout()
fig.savefig("figures/image_to_clim_soil.png")
fig.show()

# %% image -> clim+soil - appendix

columns = []
predictor_maes = []
predictor_r2s = []

for pred_name, pred_paths in predictor_paths.items():
    for pred_path in pred_paths:
        result = json.loads((pred_path / "results.json").read_text())
        if not columns: columns = list(result["mae"]["each"].keys())
        predictor_maes.append([pred_name] + list(result["mae"]["each"].values()))
        predictor_r2s.append([pred_name] + list(result["r2"]["each"].values()))

columns = [c if c.startswith("prec_") or c.startswith("temp_") else f"soil_{c}" for c in columns]
columns = [c.replace("prec_", "precipitation_").replace("temp_", "temperature_").replace("_", " ") for c in columns]
columns = ["encoder"] + columns

predictor_maes = pd.DataFrame(predictor_maes, columns=columns)
predictor_r2s = pd.DataFrame(predictor_r2s, columns=columns)

predictor_maes \
    = predictor_maes.groupby("encoder").mean().map(lambda x: rf"${x:.2f}") \
      + predictor_maes.groupby("encoder").std().fillna(0.0).map(lambda x: rf"\pm{x:.2f}$")

predictor_r2s \
    = predictor_r2s.groupby("encoder").mean().map(lambda x: rf"${x:.2f}") \
      + predictor_r2s.groupby("encoder").std().fillna(0.0).map(lambda x: rf"\pm{x:.2f}$")

predictor_mae_latex = predictor_maes.rename_axis(None).transpose().to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n")

Path(f"tables/predictor_mae.tex").write_text(predictor_mae_latex)

predictor_r2s_latex = predictor_r2s.rename_axis(None).transpose().to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n")

Path(f"tables/predictor_r2.tex").write_text(predictor_r2s_latex)
