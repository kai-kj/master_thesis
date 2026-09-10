# %%

from pathlib import Path
from typing import Sized, Any

from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_rgb
import pandas as pd
import re
import json

from scripts.colors import palette

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# %%

encoder_paths = {
    "elevation": [
        Path("models/mlp_elev_split_0_20260812_120843"),
        Path("models/mlp_elev_split_1_20260812_123150"),
        Path("models/mlp_elev_split_2_20260812_124054"),
        Path("models/mlp_elev_split_3_20260812_125005"),
        Path("models/mlp_elev_split_4_20260812_125914"),
    ],
    "coordinates": [
        Path("models/mlp_coord_split_0_20260810_184408"),
        Path("models/mlp_coord_split_1_20260812_102752"),
        Path("models/mlp_coord_split_2_20260812_104003"),
        Path("models/mlp_coord_split_3_20260812_105240"),
        Path("models/mlp_coord_split_4_20260812_110533"),
    ],
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
        Path("models/unet_split_0_20260612_122705"),
        Path("models/unet_split_1_20260721_094120"),
        Path("models/unet_split_2_20260723_091822"),
        Path("models/unet_split_3_20260820_160530"),
        Path("models/unet_split_4_20260822_063239"),
    ],
    "DeepLabV3+": [
        Path("models/deeplab_split_0_20260612_135725"),
        Path("models/deeplab_split_1_20260721_105505"),
        Path("models/deeplab_split_2_20260723_102800"),
        Path("models/deeplab_split_3_20260820_174040"),
        Path("models/deeplab_split_4_20260822_074226"),
    ],
    "Segformer": [
        Path("models/segformer_split_0_20260612_145904"),
        Path("models/segformer_split_1_20260721_115617"),
        Path("models/segformer_split_2_20260723_112827"),
        Path("models/segformer_split_3_20260821_215437"),
        Path("models/segformer_split_4_20260822_084256"),
    ],
    "TransUNet": [
        Path("models/transunet_split_0_20260612_170208"),
        Path("models/transunet_split_1_20260721_140313"),
        Path("models/transunet_split_2_20260723_133028"),
        Path("models/transunet_split_3_20260821_235725"),
        Path("models/transunet_split_4_20260822_104750"),
    ],
    "SwinFPN": [
        Path("models/swin_split_0_20260612_203922"),
        Path("models/swin_split_1_20260721_173913"),
        Path("models/swin_split_2_20260723_170627"),
        Path("models/swin_split_3_20260822_033415"),
        Path("models/swin_split_4_20260822_142831"),
    ],
}

fusion_model_paths = {
    "UNet": [
        Path("models/unet_fusion_clim_soil_split_0_20260730_125253"),
        Path("models/unet_fusion_clim_soil_split_1_20260730_000000"),
        Path("models/unet_fusion_clim_soil_split_2_20260730_000000"),
        Path("models/unet_fusion_clim_soil_split_3_20260826_113723"),
        Path("models/unet_fusion_clim_soil_split_4_20260826_124726"),
    ],
    "Segformer": [
        Path("models/segformer_fusion_clim_soil_split_0_20260730_142544"),
        Path("models/segformer_fusion_clim_soil_split_1_20260730_000000"),
        Path("models/segformer_fusion_clim_soil_split_2_20260730_000000"),
        Path("models/segformer_fusion_clim_soil_split_3_20260826_160057"),
        Path("models/segformer_fusion_clim_soil_split_4_20260826_135700"),
    ],
}

predictor_paths = {
    "frozen encoder": [
        Path("models/predictor_frozen_split_0_20260728_214343"),
        Path("models/predictor_frozen_split_1_20260728_215204"),
        Path("models/predictor_frozen_split_2_20260728_220917"),
    ],
    "trainable encoder": [
        Path("models/predictor_unfrozen_split_0_20260728_212451"),
        Path("models/predictor_unfrozen_split_1_20260728_220054"),
        Path("models/predictor_unfrozen_split_2_20260728_221759"),
    ],
}

dist_names = ["austria", "china", "czechia", "iran", "mexico", "russia", "south_africa"]

ks = [1, 5, 25, 125]


# %%

def pad(value: Any, padding: float = 0.1) -> float:
    return value + value * padding


def bp_cfg(
        n_bars: int | Sized, width: float, showfliers: bool = True,
        color: Any = "black", fill_alpha: float = 0.25
) -> dict:
    if not isinstance(n_bars, int):
        n_bars = len(n_bars)

    return {
        "widths": [width] * n_bars,
        "patch_artist": True,
        # "showmeans": True,
        "boxprops": {
            "facecolor": to_rgba(color, fill_alpha),
            "edgecolor": color,
        },
        "medianprops": {
            "color": color,
        },
        "whiskerprops": {
            "color": color,
        },
        "capprops": {
            "color": color,
        },
        "flierprops": {
            "marker": ".",
            "markerfacecolor": color,
            "markeredgecolor": color,
        },
        "showfliers": showfliers,
    }


def ttest(results: dict[str, np.ndarray], target: str, holm: bool = True) -> dict[str, float]:
    assert target in results

    model_names = [v for v in results.keys() if v != target]

    min_len = min(len(v) for v in results.values())

    uncorrected = [
        ttest_rel(results[target][:min_len], scores[:min_len])[1]
        for model_name, scores in results.items()
        if model_name in model_names
    ]

    corrected = multipletests(uncorrected, method="holm")[1]

    # return {k: float(v) for (k, _, _), v in zip(uncorrected, corrected)}

    if holm:
        return {k: float(v) for k, v in zip(model_names, corrected)}
    else:
        return {k: float(v) for k, v in zip(model_names, uncorrected)}


# %% clim-soil pretraining - main

fig, ax = plt.subplots(figsize=(8, 2.25), dpi=300)

maes = {model_name: [] for model_name in encoder_paths.keys()}
bcs = {model_name: [] for model_name in encoder_paths.keys()}

for model_name, model_paths in encoder_paths.items():
    for model_path in model_paths:
        all_results = json.loads((model_path / "results.json").read_text())
        maes[model_name] += [all_results["mae"]["average"]]
        bcs[model_name] += [all_results["bc"]["average"]]

for i, (k, vs) in enumerate(maes.items()):
    base_color = {
        "elevation": "dimgray",
        "coordinates": "dimgray",
        "climate": palette(2, 0, darken_factor=0.3),
        "soil": palette(2, 0, darken_factor=0.3),
        "climate+soil": palette(2, 1, darken_factor=0.3),
    }[k]

    ax.boxplot(vs, positions=[i], **bp_cfg(1, 0.25, color=base_color))

    labels = f"mean: {np.mean(vs):.03f}"
    ax.text(i, pad(max(vs), 0.025), labels, color=base_color, horizontalalignment="center")

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(pad(min(min(v) for v in maes.values()), -0.05), pad(max(max(v) for v in maes.values()), 0.05))
ax.set_ylabel("MAE")
ax.set_xticks(range(len(maes)), labels=maes.keys())

fig.tight_layout()
fig.savefig("figures/pretraining_mae_bc.png")
fig.show()

# %% clim-soil pretraining - ttest

print("\nPre-training, climate+soil <-> climate")

for model_name, p in \
        ttest({k: v for k, v in maes.items() if k in ["climate+soil", "climate"]}, "climate+soil", False).items():
    print(f"- {model_name}: {p:.4f}")

print("\nPre-training, climate <-> soil")

for model_name, p in \
        ttest({k: v for k, v in maes.items() if k in ["climate", "soil"]}, "climate", False).items():
    print(f"- {model_name}: {p:.4f}")

print("\nPre-training, climate+soil <-> soil")

for model_name, p in \
        ttest({k: v for k, v in maes.items() if k in ["climate+soil", "soil"]}, "climate+soil", False).items():
    print(f"- {model_name}: {p:.4f}")

print("\nPre-training, climate <-> coordinates")

for model_name, p in \
        ttest({k: v for k, v in maes.items() if k in ["climate", "coordinates"]}, "climate", False).items():
    print(f"- {model_name}: {p:.4f}")

print("\nPre-training, climate+soil <-> coordinates")

for model_name, p in \
        ttest({k: v for k, v in maes.items() if k in ["climate+soil", "coordinates"]}, "climate+soil", False).items():
    print(f"- {model_name}: {p:.4f}")

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

# maes_mean["mean"] = {k: np.mean(np.mean(v, axis=1)) for k, v in maes.items()}
# maes_std["mean"] = {k: np.std(np.mean(v, axis=1), ddof=1) for k, v in maes.items()}
# bcs_mean["mean"] = {k: np.mean(np.mean(v, axis=1)) for k, v in bcs.items()}
# bcs_std["mean"] = {k: np.std(np.mean(v, axis=1), ddof=1) for k, v in bcs.items()}

maes_table = maes_mean.astype(str)
bcs_table = bcs_mean.astype(str)

for col in maes_table.columns:
    maes_table[col] = [rf"${m:.2f}\pm{s:.2f}$" for m, s in zip(maes_mean[col], maes_std[col])]

for col in bcs_table.columns:
    bcs_table[col] = [rf"${m:.2f}\pm{s:.2f}$" for m, s in zip(bcs_mean[col], bcs_std[col])]

mae_bc_table = pd.concat({"MAE": maes_table.transpose(), "Bray-Curtis": bcs_table.transpose()}, names=[""])

mae_bc_table_latex = mae_bc_table \
    .rename_axis(["metric", "genus"]) \
    .to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n")

mae_bc_table_latex = re.sub(r"\n?\\cline{\d+-\d+}", "", mae_bc_table_latex)

Path(f"tables/encoder_mae_bc.tex").write_text(mae_bc_table_latex)

# %%

metric = "f1"
agg = "weighted"

all_results = []

for base_mode_name, base_model_paths in core_model_paths.items():
    for split, base_model_path in enumerate(base_model_paths):
        result = pd.read_csv(base_model_path / f"metrics_id.csv").set_index("metric")
        all_results.append([base_mode_name, "id", 0, split, result.loc[agg, metric]])

        for dist in dist_names:
            if not (base_model_path / f"metrics_{dist}.csv").exists():
                continue

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

best_scores_dist = all_results[["dist", "score"]].groupby("dist").max()
best_scores_dist = best_scores_dist.rename(columns={"score": "best_score_dist"}).reset_index()  # type: ignore

best_scores_model = all_results[["model", "score"]].groupby("model").max()
best_scores_model = best_scores_model.rename(columns={"score": "best_score_model"}).reset_index()  # type: ignore

all_results = pd.merge(all_results, best_scores_dist, on="dist", how="left")
all_results["rel_score_dist"] = all_results["score"] / all_results["best_score_dist"]
all_results = all_results.drop(columns=["best_score_dist"])

all_results = pd.merge(all_results, best_scores_model, on="model", how="left")
all_results["rel_score_model"] = all_results["score"] / all_results["best_score_model"]
all_results = all_results.drop(columns=["best_score_model"])

all_results = all_results[~((all_results["dist"] != "id") & (all_results["k"] == 0))]

# %% core performance - main

core_results = {k: g["score"].values for k, g in all_results.loc[all_results["dist"] == "id"].groupby("model")}
core_results = {k: core_results[k] for k in core_model_paths.keys()}

fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)

# ax.boxplot(core_results.values(), tick_labels=core_results.keys(), **bp_cfg(core_results.keys(), 0.25))  # type: ignore

for i, (k, vs) in enumerate(core_results.items()):
    base_color = palette(len(core_results), i, darken_factor=0.3)
    ax.boxplot(vs, positions=[i], **bp_cfg(1, 0.25, color=base_color))
    labels = f"mean: {np.mean(vs):.03f}"  # type: ignore
    ax.text(i, pad(max(vs), 0.0025), labels, color=base_color, horizontalalignment="center")

# ax.set_ylim(0.75, 0.9)
ax.set_ylabel(f"{agg} {metric} score")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(range(len(core_results)), labels=core_results.keys())

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
        formatted[col] = [rf"${m:.2f}\pm{s:.2f}$" for m, s in zip(means[col], stds[col])]  # type: ignore

    core_model_perf_table[base_model_name] = formatted

core_model_perf_table = pd.concat(core_model_perf_table, names=["model", "metric"])

# core_model_perf_table_latex = core_model_perf_table \
#     .to_latex(escape=False) \
#     .replace("\n\\toprule", "") \
#     .replace("\n\\midrule", "\n\\hline") \
#     .replace("\n\\bottomrule", "") \
#     .replace("\\cline{1-6}", "") \
#     .replace("\n ", "\n")
#
# Path(f"tables/core.tex").write_text(core_model_perf_table_latex)

# %%

print("\nCore, Segformer")

for model_name, p in ttest(core_results, "Segformer").items():
    print(f"- {model_name}: {p:.4f}")

print("\nCore, TransUNet")

for model_name, p in ttest(core_results, "TransUNet").items():
    print(f"- {model_name}: {p:.4f}")

# print(holm_ttest(core_results, "Segformer"))
# print(holm_ttest(core_results, "TransUNet"))

# %% tuning - by model - main

fig, ax = plt.subplots(figsize=(8, 2.5), dpi=300)

mean_by_k = {k: [] for k in ks}

for i, k in enumerate(ks):
    ax.text(i * (len(core_model_paths) + 1) + 2, 1.025, f"k = {k}", horizontalalignment="center")

    for j, (base_mode_name, base_model_path) in enumerate(core_model_paths.items()):
        result = all_results.loc[(all_results["model"] == base_mode_name) & (all_results["k"] == k), "rel_score_dist"]
        result = result.drop([result.idxmax(), result.idxmin()])

        pos = i * (len(core_model_paths) + 1) + j

        base_color = palette(len(core_model_paths), j, darken_factor=0.3)
        ax.boxplot(result, positions=[pos], **bp_cfg(1, 0.5, color=base_color))
        # mean_by_k[k].append((pos, result.mean().item()))
        mean_by_k[k].append((pos, np.mean(sorted(list(result))[1:-1]).item()))

for k, vs in mean_by_k.items():
    ax.plot([pos for pos, _ in vs], [best for _, best in vs], color="gray", linestyle=":")

    best = max(v[1] for v in vs)

    for pos, v in vs:
        # ax.scatter(pos, v, color="red" if v == best else "orange", marker="^", s=50, zorder=10)
        ax.scatter(pos, v, color="crimson", marker="^", s=30, zorder=10, label="mean")

x_tick_locations = [j * (len(core_model_paths) + 1) + i for j in range(len(ks)) for i in
                    range(len(core_model_paths))]
ax.set_xticks(x_tick_locations, [m[0] for m in core_model_paths.items()] * (len(ks)), rotation=45)
ax.vlines(x_tick_locations, 0, 1.0, linestyles=":", color="lightgray", zorder=0)

plt.setp(ax.get_xticklabels(), ha="right")

ax.set_ylim(pad(all_results["rel_score_dist"].quantile(0.05), -0.04), pad(1.0, 0.04))
ax.set_ylabel(f"relative {agg} {metric} score")
ax.spines[["top", "right"]].set_visible(False)
# ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig("figures/tuning_by_model.png")
fig.show()

# %% tuning - by site - main

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(8, 2.4),
    dpi=300,
    gridspec_kw={
        "height_ratios": [2, 1],
        "hspace": 0.1,
    },
)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

mean_by_k = {k: [] for k in ks}

for i, k in enumerate(ks):
    x_center = i * (len(dist_names) + 1) + 3

    ax_top.text(x_center, 1.01, f"k = {k}", horizontalalignment="center")

    for j, dist_name in enumerate(dist_names):
        result = all_results.loc[(all_results["dist"] == dist_name) & (all_results["k"] == k), "rel_score_model"]

        if len(result) == 0:
            continue

        pos = i * (len(dist_names) + 1) + j
        base_color = palette(len(dist_names), j, darken_factor=0.3)

        for ax in (ax_top, ax_bottom):
            ax.boxplot(result, positions=[pos], **bp_cfg(1, 0.5, color=base_color))

        mean_by_k[k].append((pos, result.mean().item()))

x_tick_locations = [j * (len(dist_names) + 1) + i for j in range(len(ks)) for i in range(len(dist_names))]

x_tick_labels = [d.replace("_", " ").title() for d in dist_names] * len(ks)

ax_top.set_ylim(0.55, 1.0 + 0.02)
ax_bottom.set_ylim(0.0, 0.25)

ax_bottom.set_xticks(x_tick_locations, x_tick_labels, rotation=45)
plt.setp(ax_bottom.get_xticklabels(), ha="right")

ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

for ax in (ax_top, ax_bottom):
    ymin, ymax = ax.get_ylim()
    ax.vlines(x_tick_locations, ymin, ymax, linestyles=":", color="lightgray", zorder=0)

ax_top.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax_bottom.set_yticks([0.0, 0.1, 0.2])

ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)

ax_top.spines[["top", "right"]].set_visible(False)
ax_bottom.spines[["right"]].set_visible(False)

fig.supylabel(f"relative {agg} {metric} score", x=0.01, fontsize=8, va="center")

fig.subplots_adjust(
    left=0.07,
    right=0.99,
    top=0.93,
    bottom=0.25,
    hspace=0.025,
)

top_pos = ax_top.get_position()
bottom_pos = ax_bottom.get_position()

kwargs = dict(transform=fig.transFigure, color="black", clip_on=False, linewidth=0.8)
ax_top.plot([top_pos.x0 - 0.008, top_pos.x0 + 0.008], [top_pos.y0 - 0.008, top_pos.y0 + 0.008], **kwargs)
ax_bottom.plot([top_pos.x0 - 0.008, top_pos.x0 + 0.008], [bottom_pos.y1 - 0.008, bottom_pos.y1 + 0.008], **kwargs)

fig.savefig("figures/tuning_by_site.png")
fig.show()

# %% tuning - ttest

# {r["model"]: r["score"] for _, r in all_results[all_results["k"] == 0].iterrows()}


print("\nFT, k = 0, DeepLabV3+")

res = ttest(
    all_results[all_results["k"] == 0].groupby("model")["rel_score_dist"].apply(np.array).to_dict(),
    "DeepLabV3+",
)

for model_name, p in res.items():
    print(f"- {model_name}: {p:.4f}")

print("\nFT, k = 5, UNet")

res = ttest(
    all_results[all_results["k"] == 5].groupby("model")["rel_score_dist"].apply(np.array).to_dict(),
    "UNet",
)

for model_name, p in res.items():
    print(f"- {model_name}: {p:.4f}")

print("\nFT, k = 125, Segformer")

res = ttest(
    all_results[all_results["k"] == 125].groupby("model")["score"].apply(np.array).to_dict(),
    "Segformer",

)

for model_name, p in res.items():
    print(f"- {model_name}: {p:.4f}")

print("\nFT, Austria")

for k in [1, 5, 25, 125]:
    res = ttest(
        all_results[all_results["k"] == k].groupby("dist")["rel_score_model"].apply(np.array).to_dict(),
        "austria",
    )

    res = [v for k, v in res.items() if k != "south_africa"]

    print(f"k = {k}: {np.max(res)}")

print("\nFT, South Africa")

for k in [1, 5, 25, 125]:
    res = ttest(
        all_results[all_results["k"] == k].groupby("dist")["rel_score_model"].apply(np.array).to_dict(),
        "south_africa",
    )

    res = [v for k, v in res.items() if k != "austria"]

    print(f"k = {k}: {np.max(res)}")

# %% tuning - matrix

print(all_results[["dist", "k"]].groupby("dist").max())

all_results_filt = all_results.loc[all_results["dist"] != "id"].copy()

all_results_filt = all_results_filt.loc[
    all_results_filt["k"].eq(all_results_filt.groupby("dist")["k"].transform("max"))
]

all_results_mean = all_results_filt.pivot_table(
    index="dist",
    columns="model",
    values="score",
    aggfunc="mean",
)

all_results_std = all_results_filt.pivot_table(
    index="dist",
    columns="model",
    values="score",
    aggfunc="std",
)

all_results_fmt = "$" + \
                  all_results_mean.map(lambda x: f"{x:.3f}") + \
                  r"\pm" + \
                  all_results_std.map(lambda x: f"{x:.3f}") + \
                  "$"

all_results_fmt.index.name = None
all_results_fmt.columns.name = None

all_results_fmt.index = [dist.replace("_", " ").title() for dist in all_results_fmt.index]

all_results_fmt_latex = all_results_fmt.to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\\cline{1-6}", "") \
    .replace("\n ", "\n")

Path(f"tables/tune_matrix.tex").write_text(all_results_fmt_latex)

# %% tuning - appendix

tuning_perf_table_raw = []

for base_mode_name, base_model_paths in core_model_paths.items():
    for split, base_model_path in enumerate(base_model_paths):
        for dist in dist_names:
            if not (base_model_path / f"metrics_{dist}.csv").exists():
                continue

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
tuning_perf_table_raw = tuning_perf_table_raw[tuning_perf_table_raw["k"].isin(ks)]

tuning_perf_table_mean = tuning_perf_table_raw \
    .groupby(["dist", "model", "k"], sort=False)[["precision", "recall", "f1", "iou"]] \
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

# fig, ax = plt.subplots(figsize=(4, 2))
#
# mae_max = 0
#
# for i, (pred_name, pred_paths) in enumerate(predictor_paths.items()):
#     maes = []
#     r2s = []
#
#     for pred_path in pred_paths:
#         result = json.loads((pred_path / "results.json").read_text())
#         maes.append(result["mae"]["average"])
#         mae_max = max(mae_max, result["mae"]["average"])
#
#     ax.boxplot(maes, positions=[i], **bp_cfg(1, 0.25))
#
# ax.spines[["top", "right"]].set_visible(False)
# ax.set_ylim(0, pad(mae_max, 0.25))
# ax.set_ylabel("mean MAE")
# ax.set_xticks(range(len(predictor_paths)), predictor_paths.keys())
#
# # ax[1].spines[["top", "right"]].set_visible(False)
# # ax[1].set_ylim(pad(r2_min, -0.05), 1)
# # ax[1].set_ylabel("$R^2$")
# # ax[1].set_xticks(range(len(predictor_paths)), predictor_paths.keys())
#
# fig.tight_layout()
# fig.savefig("figures/image_to_clim_soil.png")
# fig.show()

fig, ax = plt.subplots(figsize=(8, 3), dpi=300)

labels_x = []
labels_t = []
labels_order = []

results = {}

for i, (pred_name, pred_paths) in enumerate(predictor_paths.items()):
    maes = {}

    for pred_path in pred_paths:
        result = json.loads((pred_path / "results.json").read_text())

        for k, v in result["mae"]["each"].items():
            if k not in maes: maes[k] = []
            maes[k].append(v)

    if labels_order == []:
        labels_order = sorted(maes.keys(), key=lambda k: np.mean(maes[k]), reverse=True)

    maes = {k: maes[k] for k in labels_order}
    base_color = palette(len(predictor_paths), i, darken_factor=0.3)

    for j, (k, v) in enumerate(maes.items()):
        pos = i * (len(maes) + 5) + j

        labels_x.append(pos)
        labels_t.append(k)
        ax.boxplot(v, positions=[pos], **bp_cfg(1, 0.5, color=base_color))

    if i == 0:
        ax.boxplot(
            np.array(list(maes.values())).flatten(), positions=[len(maes) + 0.5],
            **bp_cfg(1, 1, color=base_color)
        )
        labels_x.append(len(maes) + 0.5)
        labels_t.append("all")
    else:
        ax.boxplot(
            np.array(list(maes.values())).flatten(), positions=[len(maes) + 3.5],
            **bp_cfg(1, 1, color=base_color)
        )
        labels_x.append(len(maes) + 3 + 0.5)
        labels_t.append("all")

    results[pred_name] = np.array(list(maes.values())).flatten()

    title_text = f"{pred_name}\nmean: {np.mean(list(maes.values())):.02f}"
    ax.text(i * (len(maes) + 5) + len(maes) // 2, 0.425, title_text, color=base_color, horizontalalignment="center")

    labels_order.reverse()

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("MAE")
ax.set_xticks(labels_x, labels_t, rotation=90)

for tick in ax.get_xticklabels():
    if tick.get_text() == "all":
        tick.set_fontweight("bold")

fig.tight_layout()
fig.savefig("figures/image_to_clim_soil.png")
fig.show()

# %% ttest

print(ttest(results, "trainable encoder"))

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

predictor_table = pd.concat(
    {"MAE": predictor_maes.rename_axis(None).transpose(), "$R^2$": predictor_r2s.rename_axis(None).transpose()},
    names=[""]
)

predictor_table_latex = predictor_table \
    .rename_axis(["metric", "variable"]) \
    .to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n")

predictor_table_latex = re.sub(r"\n?\\cline{\d+-\d+}", "", predictor_table_latex)

Path(f"tables/predictor_mae_r2.tex").write_text(predictor_table_latex)

# predictor_mae_latex = predictor_maes.rename_axis(None).transpose().to_latex(escape=False) \
#     .replace("\n\\toprule", "") \
#     .replace("\n\\midrule", "\n\\hline") \
#     .replace("\n\\bottomrule", "") \
#     .replace("\n ", "\n")
#
# Path(f"tables/predictor_mae.tex").write_text(predictor_mae_latex)
#
# predictor_r2s_latex = predictor_r2s.rename_axis(None).transpose().to_latex(escape=False) \
#     .replace("\n\\toprule", "") \
#     .replace("\n\\midrule", "\n\\hline") \
#     .replace("\n\\bottomrule", "") \
#     .replace("\n ", "\n")

# Path(f"tables/predictor_r2.tex").write_text(predictor_r2s_latex)

# %% fusion - table

fusion_perf_table_raw = []

for i, (base_mode_name, base_model_paths) in enumerate(fusion_model_paths.items()):
    for split, base_model_path in enumerate(base_model_paths):
        result_id = pd.read_csv(base_model_path / f"metrics_id.csv")
        result_id[["split", "model", "dist", "k", "rand"]] = [split, base_mode_name, "id", 0, False]
        fusion_perf_table_raw.append(result_id)

        result_id_rand = pd.read_csv(base_model_path / f"metrics_id_rand.csv")
        result_id_rand[["split", "model", "dist", "k", "rand"]] = [split, base_mode_name, "id", 0, True]
        fusion_perf_table_raw.append(result_id_rand)

        for dist in dist_names:
            result = pd.read_csv(base_model_path / f"metrics_{dist}.csv")
            result[["split", "model", "dist", "k", "rand"]] = [split, base_mode_name, dist, 0, False]
            fusion_perf_table_raw.append(result)

            result_rand = pd.read_csv(base_model_path / f"metrics_{dist}.csv")
            result_rand[["split", "model", "dist", "k", "rand"]] = [split, base_mode_name, dist, 0, True]
            fusion_perf_table_raw.append(result_rand)

        for dist in dist_names:
            for k in ks:
                model_name = f"{base_model_path.name}_{dist}_{k}_shot"
                model_path = list(base_model_path.parent.glob(f"{model_name}_*"))

                if len(model_path) > 0:
                    result = pd.read_csv(model_path[0] / f"metrics_{dist}.csv")
                    result[["split", "model", "dist", "k", "rand"]] = [split, base_mode_name, dist, k, False]
                    fusion_perf_table_raw.append(result)

                    result_rand = pd.read_csv(model_path[0] / f"metrics_{dist}_rand.csv")
                    result_rand[["split", "model", "dist", "k", "rand"]] = [split, base_mode_name, dist, k, True]
                    fusion_perf_table_raw.append(result_rand)

fusion_perf_table_raw = pd.concat(fusion_perf_table_raw).reset_index(drop=True)
fusion_perf_table_raw = fusion_perf_table_raw[fusion_perf_table_raw["metric"] == "weighted"].drop(columns=["metric"])
fusion_perf_table_raw = fusion_perf_table_raw[
    fusion_perf_table_raw["k"].isin(ks) | (fusion_perf_table_raw["dist"] == "id")]

# %% fusion - main

fusion_cfgs = [
    (0, 1, "UNet", True, False),
    (0, 2, "UNet", True, True),
    (0, 5, "UNet", False, False),
    (0, 6, "UNet", False, True),
    (1, 1, "Segformer", True, False),
    (1, 2, "Segformer", True, True),
    (1, 5, "Segformer", False, False),
    (1, 6, "Segformer", False, True),
]

fig_a, ax_a = plt.subplots(figsize=(3.8, 3), dpi=300)
fig_b, ax_b = plt.subplots(figsize=(3.8, 3), dpi=300)

ax = [ax_a, ax_b]

core_color = palette(3, 2, darken_factor=0.3)

unet_scores = tuning_perf_table_raw[(tuning_perf_table_raw["model"] == "UNet")]
unet_scores = unet_scores.loc[(unet_scores["k"] == 25) | (unet_scores["k"] == 25)]

ax[0].boxplot(core_results["UNet"], positions=[0], **bp_cfg(1, 0.25, color=core_color, showfliers=False))
ax[0].boxplot(unet_scores["f1"], positions=[4], **bp_cfg(1, 0.25, color=core_color, showfliers=False))

seg_scores = tuning_perf_table_raw[(tuning_perf_table_raw["model"] == "UNet")]
seg_scores = seg_scores.loc[(seg_scores["k"] == 25) | (seg_scores["k"] == 25)]

ax[1].boxplot(core_results["Segformer"], positions=[0], **bp_cfg(1, 0.25, color=core_color, showfliers=False))
ax[1].boxplot(seg_scores["f1"], positions=[4], **bp_cfg(1, 0.25, color=core_color, showfliers=False))

for ax_i, ax_x, model_name, is_id, rand_state in fusion_cfgs:
    scores = fusion_perf_table_raw[
        (fusion_perf_table_raw["model"] == model_name) &
        (fusion_perf_table_raw["rand"] == rand_state)
        ]

    scores = scores[scores["dist"] == "id"] if is_id else scores[scores["dist"] != "id"]
    scores = scores.loc[(scores["k"] == 25) | (scores["k"] == 25)] if not is_id else scores

    base_color = palette(3, int(rand_state), darken_factor=0.3)

    ax[ax_i].boxplot(scores["f1"], positions=[ax_x], **bp_cfg(1, 0.25, color=base_color, showfliers=False))

    # if not rand_state:
    mb = scores["f1"] if is_id else np.sort(scores["f1"])[2:]
    ma = core_results[model_name] if is_id else \
        np.sort(unet_scores["f1"])[2:] if model_name == "UNet" else np.sort(seg_scores["f1"])[2:]

    delta = np.mean(mb) - np.mean(ma)

    ax[ax_i].text(ax_x, pad(np.max(scores["f1"]), 0.03), f"{delta:+.03f}", horizontalalignment="center")

    orig = ax_x - 2 if rand_state else ax_x - 1

    ax[ax_i].plot([orig, ax_x], [np.mean(ma), np.mean(mb)], color="gray", linestyle=":")
    ax[ax_i].scatter([orig, ax_x], [np.mean(ma), np.mean(mb)], color="crimson", marker="^", s=30, zorder=10)

for ax_i in [0, 1]:
    ax[ax_i].text(1, 1.1, "core performance", horizontalalignment="center")
    ax[ax_i].text(5, 1.1, "fine-tuning performance", horizontalalignment="center")

    ax[ax_i].text(1, 1.1, "core performance", horizontalalignment="center")
    ax[ax_i].text(5, 1.1, "fine-tuning performance", horizontalalignment="center")

    ax[ax_i].set_xticks(
        [0, 1, 2, 4, 5, 6],
        [
            "without fusion", "with fusion\ncorrect vars.", "with fusion\nrandom vars.",
            "without fusion", "with fusion\ncorrect vars.", "with fusion\nrandom vars."
        ],
        rotation=60,
    )

ax[0].spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("weighted F1 score")

ax[1].spines[["top", "right"]].set_visible(False)
ax[1].set_ylabel("weighted F1 score")

ax[0].set_ylim(ax[0].get_ylim()[0], 1.09)
ax[1].set_ylim(ax[1].get_ylim()[0], 1.09)

fig_a.tight_layout()
fig_a.savefig("figures/fusion_summary_a.png")
fig_a.show()

fig_b.tight_layout()
fig_b.savefig("figures/fusion_summary_b.png")
fig_b.show()

# %% appendix core tables

core_fusion_model_perf_table = []

for base_model_name, base_model_paths in fusion_model_paths.items():
    for r in [False, True]:
        split_results: list[pd.DataFrame] = []

        for base_model_path in base_model_paths:
            file_name = "metrics_id_rand.csv" if r else "metrics_id.csv"
            results = pd.read_csv(base_model_path / file_name)
            results.insert(0, "fusion", "random" if r else "correct")
            results.insert(1, "model", base_model_name)
            core_fusion_model_perf_table.append(results)

        # split_results = pd.concat(split_results)
        #
        # means = split_results
        # stds = pd.concat(split_results).groupby(level=0).std(ddof=1).fillna(0.0)
        #
        # formatted = means.astype(str)  # type: ignore
        #
        # for col in formatted.columns:
        #     formatted[col] = [rf"${m:.3f}\pm{s:.3f}$" for m, s in zip(means[col], stds[col])]  # type: ignore
        #
        # core_fusion_model_perf_table[base_model_name] = formatted

core_fusion_model_perf_table = pd.concat(core_fusion_model_perf_table).reset_index(drop=True)

g_cols = ["fusion", "model", "metric"]
v_cols = ["precision", "recall", "f1", "iou"]

core_fusion_model_perf_table \
    = core_fusion_model_perf_table.groupby(g_cols, sort=False)[v_cols].mean().map(lambda x: rf"${x:.2f}") \
      + core_fusion_model_perf_table.groupby(g_cols, sort=False)[v_cols].std().fillna(0.0).map(
    lambda x: rf"\pm{x:.2f}$")

appendix_core_table = core_model_perf_table.copy()

appendix_core_table["fusion"] = "none"
appendix_core_table = appendix_core_table.set_index("fusion", append=True)
appendix_core_table = appendix_core_table.reorder_levels(["fusion", "model", "metric"])

appendix_core_table = pd.concat([appendix_core_table, core_fusion_model_perf_table])

appendix_core_table = appendix_core_table.iloc[
    pd.Series(
        pd.Categorical(
            appendix_core_table.index.get_level_values("fusion"),
            categories=["none", "correct", "random"],
            ordered=True,
        ).codes
    ).sort_values(kind="stable").index
]

appendix_core_table_latex = appendix_core_table.to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n") \
    .replace("f1", "F1 score") \
    .replace("iou", "IoU score")

appendix_core_table_latex = re.sub(r"\n?\\cline{\d+-\d+}", "", appendix_core_table_latex)

Path(f"tables/appendix_core.tex").write_text(appendix_core_table_latex)

# %% appendix tune tables

appendix_tune_table_a = tuning_perf_table_raw.copy()
appendix_tune_table_a["fusion"] = "none"

appendix_tune_table_b = fusion_perf_table_raw.copy()
appendix_tune_table_b = appendix_tune_table_b[appendix_tune_table_b["dist"] != "id"]
appendix_tune_table_b = appendix_tune_table_b.rename(columns={"rand": "fusion"})
appendix_tune_table_b["fusion"] = appendix_tune_table_b["fusion"].apply(lambda x: "random" if x else "correct")

appendix_tune_table = pd.concat([appendix_tune_table_a, appendix_tune_table_b])
appendix_tune_table = appendix_tune_table.rename(columns={"dist": "site"})
appendix_tune_table["site"] = appendix_tune_table["site"].apply(lambda s: s.replace("_", " ").title())

appendix_tune_table["fusion"] = pd.Categorical(
    appendix_tune_table["fusion"], categories=["none", "correct", "random"], ordered=True,
)

appendix_tune_table = appendix_tune_table.sort_values("fusion", kind="stable")

g_cols = ["fusion", "model", "site", "k"]
v_cols = ["precision", "recall", "f1", "iou"]

appendix_tune_table \
    = appendix_tune_table.groupby(g_cols, sort=False)[v_cols].mean().map(lambda x: rf"${x:.2f}") \
      + appendix_tune_table.groupby(g_cols, sort=False)[v_cols].std().fillna(0.0).map(lambda x: rf"\pm{x:.2f}$")

appendix_tune_table_latex = appendix_tune_table.to_latex(escape=False) \
    .replace("\n\\toprule", "") \
    .replace("\n\\midrule", "\n\\hline") \
    .replace("\n\\bottomrule", "") \
    .replace("\n ", "\n") \
    .replace("f1", "F1 score") \
    .replace("iou", "IoU score")

appendix_tune_table_latex = re.sub(r"\n?\\cline{\d+-\d+}", "", appendix_tune_table_latex)
appendix_tune_table_latex = re.sub(r"\\begin{tabular}{\w*}\n?", "", appendix_tune_table_latex)
appendix_tune_table_latex = re.sub(r"\n?\\end{tabular}", "", appendix_tune_table_latex)

Path(f"tables/appendix_tune.tex").write_text(appendix_tune_table_latex)
