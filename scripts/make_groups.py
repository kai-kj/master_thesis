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

from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation

# %%

labels_map_path = Path("data/labels_map.csv")
assert labels_map_path.is_file()

countries_path = Path("/home/kai/Thesis/data/naturalearth/ne_10m_admin_0_countries.zip")
assert countries_path.is_file()

groups_path = Path("data/groups.csv")

treeai = TreeAISegmentation("/home/kai/Thesis/data/treeai_segmentation/", pre_cache=True)

# %%

labels_map = pd.read_csv(labels_map_path)
countries = gpd.read_file(countries_path)

trees_relabeled = treeai.get_trees_ext() \
    .merge(labels_map, left_on="label_id", right_on="data_label_id", how="left") \
    .drop(columns=["label_id", "label_name", "data_label_id", "data_label_name"]) \
    .rename(columns={"model_label_id": "label_id", "model_label_name": "label_name"}) \
    [["label_id", "label_name", "raster_name", "site_id", "geometry"]]

sites_ext = treeai.get_sites() \
    .sjoin(countries[["NAME", "geometry"]], how="left", predicate="intersects") \
    .drop(columns=["index_right"]) \
    .rename(columns={"NAME": "country"}) \
    [["site_id", "country", "geometry"]]

# %%

other_frac_cutoff = 0.4

by_site = pd.crosstab(trees_relabeled["site_id"], trees_relabeled["label_name"])
by_site["other_frac"] = by_site["other"] / by_site.sum(axis=1)

fig, ax = plt.subplots()

ax.hist(by_site["other_frac"], bins=np.linspace(0, 1, 25).tolist())
ax.vlines(other_frac_cutoff, ax.get_ylim()[0], ax.get_ylim()[1], linestyle="dashed", color="red")
ax.text(other_frac_cutoff, ax.get_ylim()[1], f"cutoff: {other_frac_cutoff}", ha="left", va="top", color="red")

ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel("fraction of \"other\" labels")
ax.set_ylabel("occurrences")

fig.tight_layout()
fig.show()

print(f"removing {sum(by_site["other_frac"] >= other_frac_cutoff)}/{len(by_site)} sites")
by_site = by_site[by_site["other_frac"] < other_frac_cutoff].drop(columns=["other_frac"])

# %%

site_grouping_cutoff = 0.7

id_groups = [
    0, 1, 14,  # Germany
    4, 5,  # Italy
    6, 7,  # Lithuania
    13,  # Russia
    21,  # Canada
    25,  # Netherlands
    31,  # Switzerland
]

dist_matrix = pdist(np.sqrt((by_site / by_site.sum(axis=0)).values), metric="braycurtis")
Z = linkage(dist_matrix, method="average")

groups = by_site.reset_index()[["site_id"]]
groups["cluster_id"] = fcluster(Z, t=site_grouping_cutoff, criterion="distance")
groups = groups.merge(sites_ext[["site_id", "country"]], on="site_id", how="left")
groups["group_id"] = groups["cluster_id"].apply(lambda i: f"{i}") + ", " + groups["country"]

cluster_ids = {v: i for i, v in enumerate(groups["group_id"].unique())}
groups["group_id"] = groups["group_id"].map(cluster_ids)
groups = groups[["site_id", "group_id", "country"]]

by_group = by_site \
    .merge(groups[["site_id", "group_id"]], on="site_id", how="left") \
    .drop(columns=["site_id"]) \
    .groupby("group_id") \
    .agg("sum")

ood_groups = list(set(groups["group_id"]) - set(id_groups))

by_group = by_group.loc[id_groups + ood_groups, :]
by_group_norm = by_group / by_group.sum(axis=0)

xs = np.arange(0, len(by_group.columns))

dendrogram_labels = groups.apply(
    lambda r: f"group {r['group_id']:02d}, site {r['site_id']:02d}, {r['country']}",
    axis=1
)

dendrogram_labels = [
    f"X {l}" if int(l.split(" ")[1].removeprefix("0").removesuffix(",")) in ood_groups else f"   {l}"
    for l in dendrogram_labels
]

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

for _, g in by_group.iterrows():
    data = np.array(g.tolist())
    ax1.bar(xs, data, bottom=bottom, label=f"{g.name:02d}", hatch=".." if g.name in ood_groups else "")
    bottom += data

bottom = np.zeros_like(by_group_norm.columns)

for _, g in by_group_norm.iterrows():
    data = np.array(g.tolist())
    ax2.bar(xs, data, bottom=bottom, label=f"{g.name:02d}", hatch=".." if g.name in ood_groups else "")
    bottom += data

ax1.spines[["right", "top"]].set_visible(False)
ax1.set_xticks(xs, "" * len(xs))
ax1.legend(loc="upper right", frameon=False)
ax1.set_ylabel("occurrences")

ax2.spines[["right", "top"]].set_visible(False)
ax2.set_xticks(xs, by_group.columns, rotation=90)
ax2.set_ylabel("occurrences normalized")

fig.tight_layout()
fig.savefig("figures/sites_split.png")
fig.show()

# %%

# id_countries = [
#     # "Belgium",
#     "Canada",
#     # "China",
#     # "Czechia",
#     # "Estonia",
#     # "Finland",
#     # "France",
#     # "Georgia",
#     "Germany",
#     # "Greece",
#     # "Iran",
#     # "Italy",
#     # "Japan",
#     "Lithuania",
#     # "Mexico",
#     # "Netherlands",
#     # "Poland",
#     # "Portugal",
#     "Russia",
#     # "Slovakia",
#     # "Spain",
#     # "Sweden",
#     "Switzerland",
#     # "Turkey",
# ]
#
# by_country = by_site \
#     .reset_index() \
#     .merge(sites_ext[["site_id", "country"]], on="site_id", how="left") \
#     .drop(columns=["site_id"]) \
#     .groupby("country") \
#     .agg("sum") \
#     .reset_index()
#
# by_country["id"] = by_country["country"].apply(lambda c: c not in id_countries)
# by_country = by_country.sort_values("id").drop(columns=["id"])
#
# by_country_norm = pd.DataFrame(by_country.iloc[:, 1:] / by_country.iloc[:, 1:].sum(axis=0))
# by_country_norm.insert(0, "country", by_country["country"])
#
# fig, ax = plt.subplots(1, 2, figsize=(16, 8))
#
# ax[0].set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(by_country.shape[0], pastel_factor=0.5, rng=0)]))
# ax[1].set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(by_country.shape[0], pastel_factor=0.5, rng=0)]))
#
# xs = np.arange(by_country.shape[1] - 1)
#
# bottom_0 = np.zeros_like(xs, dtype=np.float64)
# bottom_1 = np.zeros_like(xs, dtype=np.int64)
#
# for _, row in by_country_norm.iterrows():
#     row = row.tolist()
#     ax[0].bar(xs, row[1:], bottom=bottom_0, hatch=".." if row[0] in id_countries else None, label=row[0])
#     bottom_0 += row[1:]
#
# for _, row in by_country.iterrows():
#     row = row.tolist()
#     ax[1].bar(xs, row[1:], bottom=bottom_1, hatch=".." if row[0] in id_countries else None, label=row[0])
#     bottom_1 += row[1:]
#
# ax[0].set_xticks(xs, labels=by_country.columns[1:], rotation=90)
# ax[0].spines[["right", "top"]].set_visible(False)
# ax[0].set_ylabel("normlized count")
#
# ax[1].set_xticks(xs, labels=by_country.columns[1:], rotation=90)
# ax[1].spines[["right", "top"]].set_visible(False)
# ax[1].set_ylabel("count")
# ax[1].legend(loc="upper right")
#
# fig.tight_layout()
# fig.show()

# %%

groups_split = groups.copy()

groups_split["dist"] = "none"
groups_split.loc[groups["group_id"].isin(id_groups), "dist"] = "ID"
groups_split.loc[groups["group_id"].isin(ood_groups), "dist"] = "OOD"

print("ID countries:")

for c in groups_split[groups_split["dist"] == "ID"]["country"].unique():
    print(f"- {c}")

print("OOD countries:")

for c in groups_split[groups_split["dist"] == "OOD"]["country"].unique():
    print(f"- {c}")

groups_split = groups_split.drop(columns=["country"])

# %%

groups_path.parent.mkdir(parents=True, exist_ok=True)
groups_split.to_csv(groups_path, index=False)
