# %% imports

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from cycler import cycler
from distinctipy import get_colors, get_hex
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from matplotlib.gridspec import GridSpec

from scripts.check_input_outputs import check_input_outputs

# %%

inputs: dict[str, Path] = {
    "sites": Path("data/intermediate/sites_a.gpkg"),
    "trees_a": Path("data/intermediate/trees_a.gpkg"),
    "labels": Path("data/intermediate/labels.csv"),
    "labels_map": Path("data/intermediate/labels_map.csv"),
}

outputs: dict[str, Path] = {
    "groups": Path("data/intermediate/groups.csv"),
    "sites_split_figure": Path("figures/sites_split.png"),
}

check_input_outputs(inputs, outputs)

# %%

sites = gpd.read_file(inputs["sites"])
trees = gpd.read_file(inputs["trees_a"])
labels = pd.read_csv(inputs["labels"])
labels_map = pd.read_csv(inputs["labels_map"])

# %%

trees_labeled = trees \
    .merge(labels_map, on="tree_id") \
    [["label_id", "label_name", "site_id", "geometry"]]

by_site = pd \
    .crosstab(trees_labeled["site_id"], trees_labeled["label_id"]) \
    .rename(columns={r["label_id"]: r["label_name"] for _, r in labels[["label_id", "label_name"]].iterrows()})

by_site["other_frac"] = by_site["Other"] / by_site.sum(axis=1)

other_frac_cutoff = 0.3

fig, ax = plt.subplots()

ax.hist(by_site["other_frac"], bins=np.linspace(0, 1, 25).tolist())
ax.vlines(other_frac_cutoff, ax.get_ylim()[0], ax.get_ylim()[1], linestyle="dashed", color="red")
ax.text(other_frac_cutoff, ax.get_ylim()[1], f"cutoff: {other_frac_cutoff}", ha="left", va="top", color="red")

ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel("fraction \"Other\" of trees")
ax.set_ylabel("occurrences")

fig.tight_layout()
fig.show()

print(f"removing {sum(by_site["other_frac"] >= other_frac_cutoff)} / {len(by_site)} sites")

by_site = by_site[by_site["other_frac"] < other_frac_cutoff].drop(columns=["other_frac"])

# %%

site_grouping_cutoff = 0.7
ood_groups = [
    "GROUP02",  # Germany
    "GROUP03",  # Iran
    "GROUP05",  # Italy
    "GROUP07",  # Turkey
    "GROUP08",  # Spain
    "GROUP09",  # Switzerland
    "GROUP12",  # Finland
    "GROUP13",  # Japan
    "GROUP14",  # Russia
    "GROUP17",  # China
    "GROUP18",  # Portugal
    "GROUP19",  # Russia
    "GROUP20",  # Russia
    "GROUP21",  # Slovakia
    "GROUP22",  # Mexico
]

dist_matrix = pdist(np.sqrt((by_site / by_site.sum(axis=0)).values), metric="braycurtis")
Z = linkage(dist_matrix, method="average")

groups_a = by_site.reset_index()[["site_id"]]
groups_a["cluster_id"] = fcluster(Z, t=site_grouping_cutoff, criterion="distance")
groups_a["cluster_id"] = [f"CLUSTER{i:02d}" for i in groups_a["cluster_id"]]
groups_a = groups_a.merge(sites[["site_id", "country"]], on="site_id", how="left")
groups_a["group_id"] = groups_a["cluster_id"] + ", " + groups_a["country"]

cluster_ids = {v: f"GROUP{i:02d}" for i, v in enumerate(groups_a["group_id"].unique())}
groups_a["group_id"] = groups_a["group_id"].map(cluster_ids)

by_group = by_site \
    .merge(groups_a[["site_id", "group_id"]], on="site_id", how="left") \
    .drop(columns=["site_id"]) \
    .groupby("group_id") \
    .agg("sum")

id_groups = list(set(groups_a["group_id"]) - set(ood_groups))

by_group = by_group.loc[id_groups + ood_groups, :]
by_group_norm = by_group / by_group.sum(axis=0)

xs = np.arange(0, len(by_group.columns))

dendrogram_labels = (groups_a["group_id"] + " " + groups_a["site_id"] + " " + groups_a["country"]).tolist()
dendrogram_labels = [f"X {l}" if l.split(" ")[0] in ood_groups else f"   {l}" for l in dendrogram_labels]
dendrogram_labels = [l.replace("Russian Federation", "Russia") for l in dendrogram_labels]

fig = plt.figure(figsize=(16, 16))
gs = GridSpec(2, 2, figure=fig)
ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])

dendrogram(Z, labels=dendrogram_labels, orientation="left", color_threshold=0, leaf_font_size=8, ax=ax0)
ax0.vlines(site_grouping_cutoff, ax0.get_ylim()[0], ax0.get_ylim()[1], linestyle="dashed", color="red")
ax0.spines[["left", "right", "top"]].set_visible(False)
ax0.set_xlabel("average distance")

ax1.set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(len(by_group), pastel_factor=0.5, rng=0)]))
ax2.set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(len(by_group), pastel_factor=0.5, rng=0)]))

bottom = np.zeros_like(by_group.columns)

for _, country in by_group.iterrows():
    name = country.name
    data = np.array(country.tolist())
    ax1.bar(xs, data, bottom=bottom, label=name, hatch="xx" if name in ood_groups else "")
    bottom += data

bottom = np.zeros_like(by_group_norm.columns)

for _, country in by_group_norm.iterrows():
    name = country.name
    data = np.array(country.tolist())
    ax2.bar(xs, data, bottom=bottom, label=name, hatch="xx" if name in ood_groups else "")
    bottom += data

ax1.spines[["right", "top"]].set_visible(False)
ax1.set_xticks(xs, "" * len(xs))
ax1.legend(loc="upper right", frameon=False)
ax1.set_ylabel("occurrences")

ax2.spines[["right", "top"]].set_visible(False)
ax2.set_xticks(xs, by_group.columns, rotation=90)
ax2.set_ylabel("occurrences normalized")

fig.tight_layout()
fig.savefig(outputs["sites_split_figure"])
fig.show()

# %%

groups_b = groups_a[["site_id", "group_id"]]

groups_b["dist"] = "none"
groups_b.loc[groups_a["group_id"].isin(id_groups), "dist"] = "ID"
groups_b.loc[groups_a["group_id"].isin(ood_groups), "dist"] = "OOD"

# %%

groups_b.to_csv(outputs["groups"], index=False)
