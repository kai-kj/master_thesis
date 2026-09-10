# %% imports

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import numpy as np
from cycler import cycler
from distinctipy import get_colors, get_hex

from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation
from dependencies.treeai_labels.treeai_labels import load_labels as treeai_load_labels

# %% setup

countries_path = Path("/home/kai/Thesis/data/naturalearth/ne_10m_admin_0_countries.zip")
assert countries_path.is_file()

treeai = TreeAISegmentation("/home/kai/Thesis/data/treeai_segmentation/")

labels_map_path = Path("data/labels_map_africa.csv")

# %% load data

trees = treeai.get_trees()
rasters = treeai.get_rasters()
sites = treeai.get_sites()
species = treeai_load_labels()
countries = gpd.read_file(countries_path)

# %%

sites_ext = sites \
    .sjoin(countries[["NAME", "geometry"]], how="left", predicate="intersects") \
    .drop(columns=["index_right"]) \
    .rename(columns={"NAME": "country"}) \
    [["site_id", "country", "geometry"]]

genus_by_site = trees \
    .merge(species[["label_id", "genus_name"]], on="label_id") \
    .merge(rasters[["raster_name", "site_id"]], on="raster_name")

genus_by_site = pd \
    .crosstab(genus_by_site["site_id"], genus_by_site["genus_name"]) \
    .reset_index() \
    .rename_axis(None, axis=0) \
    .rename_axis(None, axis=1)

genus_by_site = sites_ext.merge(genus_by_site, on="site_id")

genus_by_country = genus_by_site \
    .drop(columns=["site_id", "geometry"]) \
    .groupby("country") \
    .agg("sum") \
    .reset_index()

# %%

fig, ax = plt.subplots(figsize=(10, 5))

ys = genus_by_country[genus_by_country["country"] == "South Africa"].iloc[:, 1:]
ys = ys.loc[:, (ys > 100).all(axis=0)]

xs = np.arange(ys.shape[1])

ax.bar(xs, ys.iloc[0, :])
ax.axhline([750], color="orange", linestyle="dashed")  # type: ignore

ax.set_xticks(xs, labels=ys.columns, rotation=90)
ax.spines[["right", "top"]].set_visible(False)
ax.set_ylabel("total count")

fig.tight_layout()
fig.show()

# %%

labels_list = [
    # "Bridelia",
    # "Burkea",
    "Combretum",
    # "Commiphora",
    # "Croton",
    "Dichrostachys",
    "Diplorhynchus",
    # "Entada",
    # "Euclea",
    "Grewia",
    # "Gymnosporia",
    # "Lannea",
    "Mundulea",
    # "Myrothamnus",
    # "Ozoroa",
    # "Pseudolachnostylis",
    "Pterocarpus",
    # "Rhoicissus",
    # "Strychnos",
    # "Terminalia",
    # "Vangueria",
    # "Vitex",
    # "Ziziphus",
]

labels_list = ["Background", "Other"] + labels_list

labels = treeai \
    .get_labels() \
    .merge(species, on="label_id") \
    [["label_id", "label_name_x", "genus_name", "genus_id"]] \
    .rename(columns={"label_id": "data_label_id", "label_name_x": "data_label_name"})

labels["model_label_name"] = labels["genus_name"].apply(lambda n: n if n in labels_list else "Other")

labels.loc[labels["data_label_name"] == "Fully labeled background", "model_label_name"] = "Background"
labels.loc[labels["data_label_name"] == "Partially labeled background", "model_label_name"] = "Background"

labels["model_label_id"] = labels["model_label_name"].apply(lambda n: labels_list.index(n))
labels["model_label_name"] = labels["model_label_name"].apply(lambda n: n.replace(" ", "_").lower())

labels = labels[["data_label_id", "data_label_name", "model_label_id", "model_label_name"]]
labels = labels.sort_values(["model_label_id", "data_label_id"])

print(f"{len(labels["model_label_name"].unique())} unique labels:")

for l in labels["model_label_name"].unique():
    print(f"- {l}")

# %%

labels_map_path.parent.mkdir(parents=True, exist_ok=True)
labels.to_csv(labels_map_path, index=False)
