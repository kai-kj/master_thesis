# %% imports

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from cycler import cycler
from distinctipy import get_colors, get_hex

from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation

# %%

labels_map_path = Path("data/labels_map.csv")
assert labels_map_path.is_file()

countries_path = Path("/home/kai/Thesis/data/naturalearth/ne_10m_admin_0_countries.zip")
assert countries_path.is_file()

provinces_path = Path("/home/kai/Thesis/data/naturalearth/ne_10m_admin_1_states_provinces.zip")
assert provinces_path.is_file()

sites_split_path = Path("data/sites_split.gpkg")

treeai = TreeAISegmentation("/home/kai/Thesis/data/treeai_segmentation/", pre_cache=True)

# %%

labels_map = pd.read_csv(labels_map_path)
countries = gpd.read_file(countries_path)
provinces = gpd.read_file(provinces_path)

trees_relabeled = treeai.get_trees_ext() \
    .merge(labels_map, left_on="label_id", right_on="data_label_id", how="left") \
    .drop(columns=["label_id", "label_name", "data_label_id", "data_label_name"]) \
    .rename(columns={"model_label_id": "label_id", "model_label_name": "label_name"}) \
    [["label_id", "label_name", "raster_name", "site_id", "geometry"]]

sites_ext = treeai.get_sites() \
    .sjoin(countries[["NAME", "geometry"]], how="left", predicate="intersects") \
    .drop(columns=["index_right"]) \
    .rename(columns={"NAME": "country"}) \
    .sjoin(provinces[["name", "geometry"]], how="left", predicate="intersects") \
    .drop(columns=["index_right"]) \
    .rename(columns={"name": "province"}) \
    [["site_id", "country", "province", "geometry"]]

# %%

other_frac_cutoff = 0.5

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

id_country_province = [
    ("Canada", "Québec"),
    ("Germany", "Baden-Württemberg"),
    # ("Spain", "Granada"),
]

sites_split = sites_ext.copy()

country_mask = sites_split.apply(lambda v: v["country"] in [r[0] for r in id_country_province], axis=1)
province_mask = sites_split.apply(lambda v: v["province"] in [r[1] for r in id_country_province], axis=1)

sites_split["dist"] = "OOD"
sites_split.loc[country_mask & province_mask, "dist"] = "ID"

print("ID countries:")

for c in sites_split[sites_split["dist"] == "ID"]["country"].unique():
    print(f"- {c}")

print("OOD countries:")

for c in sites_split[sites_split["dist"] == "OOD"]["country"].unique():
    print(f"- {c}")

sites_split = sites_split[["site_id", "dist", "country", "province", "geometry"]]

# %%

by_country = by_site \
    .reset_index() \
    .merge(sites_split[["site_id", "country", "province"]], on="site_id", how="left") \
    .drop(columns=["site_id"]) \
    .groupby(["country", "province"]) \
    .agg("sum") \
    .reset_index()

country_mask = by_country.apply(lambda v: v["country"] in [r[0] for r in id_country_province], axis=1)
province_mask = by_country.apply(lambda v: v["province"] in [r[1] for r in id_country_province], axis=1)

by_country["id"] = ~(country_mask & province_mask)
by_country = by_country.sort_values("id").drop(columns=["id"])

by_country_norm = pd.DataFrame(by_country.iloc[:, 2:] / by_country.iloc[:, 2:].sum(axis=0))
by_country_norm.insert(0, "country", by_country["country"])
by_country_norm.insert(1, "province", by_country["province"])

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(by_country.shape[0], pastel_factor=0.5, rng=0)]))
ax[1].set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(by_country.shape[0], pastel_factor=0.5, rng=0)]))

xs = np.arange(by_country.shape[1] - 2)

bottom_0 = np.zeros_like(xs, dtype=np.float64)
bottom_1 = np.zeros_like(xs, dtype=np.int64)

for _, row in by_country_norm.iterrows():
    row = row.tolist()
    is_id = row[0] in [r[0] for r in id_country_province] and row[1] in [r[1] for r in id_country_province]
    ax[0].bar(xs, row[2:], bottom=bottom_0, hatch="xx" if is_id else None, label=f"{row[0]}, {row[1]}")
    bottom_0 += row[2:]

for _, row in by_country.iterrows():
    row = row.tolist()
    is_id = row[0] in [r[0] for r in id_country_province] and row[1] in [r[1] for r in id_country_province]
    ax[1].bar(xs, row[2:], bottom=bottom_1, hatch="xx" if is_id else None, label=f"{row[0]}, {row[1]}")
    bottom_1 += row[2:]

ax[0].set_xticks(xs, labels=by_country.columns[2:], rotation=90)
ax[0].spines[["right", "top"]].set_visible(False)
ax[0].set_ylabel("normalized count")

ax[1].set_xticks(xs, labels=by_country.columns[2:], rotation=90)
ax[1].spines[["right", "top"]].set_visible(False)
ax[1].set_ylabel("count")
ax[1].legend(loc="upper right", fontsize="small", frameon=False)

fig.tight_layout()
fig.savefig("figures/id_ood_split.png")
fig.show()

# %%

sites_split_path.parent.mkdir(parents=True, exist_ok=True)
sites_split.to_file(sites_split_path)
