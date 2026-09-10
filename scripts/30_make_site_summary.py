# %%

import json
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from scripts.dataset import DatasetTreeAI
from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation

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

sites_path = Path("data/sites_split.gpkg")
assert sites_path.is_file()

label_map_main_path = Path("data/labels_map.csv")
assert label_map_main_path.is_file()

label_map_africa_path = Path("data/labels_map_africa.csv")
assert label_map_africa_path.is_file()

treeai = TreeAISegmentation("/home/kai/Thesis/data/treeai_segmentation/", pre_cache=True)

# %%

label_map = pd.concat([
    pd.read_csv(label_map_main_path),
    pd.read_csv(label_map_africa_path),
])

trees = treeai.get_trees_ext() \
    .merge(gpd.read_file(sites_path), left_on="site_id", right_on="site_id", how="left") \
    .merge(label_map, left_on="label_id", right_on="data_label_id", how="left")

# %%

dataset_names = {
    "ID": "Core",
    "OOD_china": "China",
    "OOD_czechia": "Czechia",
    "OOD_iran": "Iran",
    "OOD_mexico": "Mexico",
    "OOD_russia": "Russia",
    "OOD_austria": "Austria",
    "OOD_south_africa": "South Africa",
}

labels_main = [
    "abies",
    "acer",
    "betula",
    "fagus",
    "picea",
    "pinus",
    "populus",
    "thuja",
]

labels_africa = [
    "combretum",
    "dichrostachys",
    "diplorhynchus",
    "grewia",
    "mundulea",
    "pterocarpus",
]

# %%

fig, ax = plt.subplots(2, 4, figsize=(2 * 6, 2 * 3))

for i, dist in enumerate(dataset_names.keys()):
    stats: dict[str, int] = trees[trees["dist"] == dist]["model_label_name"].value_counts().to_dict()  # type: ignore
    labels = labels_main if dist != "OOD_south_africa" else labels_africa

    ax[i // 4, i % 4].bar(labels, [stats.get(l, 0) for l in labels])  # type: ignore
    ax[i // 4, i % 4].set_xticks(range(len(labels)), labels, rotation=90)
    ax[i // 4, i % 4].spines[["top", "right"]].set_visible(False)
    ax[i // 4, i % 4].set_ylabel("occurrences")
    ax[i // 4, i % 4].set_title(dataset_names[dist])

fig.tight_layout()
fig.savefig(f"figures/site_summary.png")
fig.show()

# %%

for i, dist in enumerate(dataset_names.keys()):
    stats: dict[str, int] = trees[trees["dist"] == dist]["model_label_name"].value_counts().to_dict()  # type: ignore
    labels = labels_main if dist != "OOD_south_africa" else labels_africa

    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)

    ax.bar(labels, [stats.get(l, 0) for l in labels])  # type: ignore
    ax.set_xticks(range(len(labels)), labels, rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("occurrences")

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(axis="y", labelrotation=90)

    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.3, top=0.9)
    fig.savefig(f"figures/site_summary_{dist}.png")
    fig.show()

# %%

for i, dist in enumerate(dataset_names.keys()):
    ds_train = DatasetTreeAI(dist, "train", 0)
    ds_test = DatasetTreeAI(dist, "test", 0)

    print(f"{dist}: #train: {len(ds_train)}, #test: {len(ds_test)}")
