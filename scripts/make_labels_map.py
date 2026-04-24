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

labels_map_path = Path("data/labels_map.csv")

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

species_by_site = trees \
    .merge(species[["label_id", "species_name"]], on="label_id") \
    .merge(rasters[["raster_name", "site_id"]], on="raster_name")

species_by_site = pd \
    .crosstab(species_by_site["site_id"], species_by_site["species_name"]) \
    .reset_index() \
    .rename_axis(None, axis=0) \
    .rename_axis(None, axis=1)

species_by_site = sites_ext.merge(species_by_site, on="site_id")

genus_by_site = trees \
    .merge(species[["label_id", "genus_name"]], on="label_id") \
    .merge(rasters[["raster_name", "site_id"]], on="raster_name")

genus_by_site = pd \
    .crosstab(genus_by_site["site_id"], genus_by_site["genus_name"]) \
    .reset_index() \
    .rename_axis(None, axis=0) \
    .rename_axis(None, axis=1)

genus_by_site = sites_ext.merge(genus_by_site, on="site_id")

# %%

species_by_country = species_by_site \
    .drop(columns=["site_id", "geometry"]) \
    .groupby("country") \
    .agg("sum") \
    .reset_index()

genus_by_country = genus_by_site \
    .drop(columns=["site_id", "geometry"]) \
    .groupby("country") \
    .agg("sum") \
    .reset_index()


# %%

def make_site_summary(sites_source: pd.DataFrame, by: str, countries: list[str]):
    sites_filt: pd.DataFrame = sites_source[sites_source["country"].isin(countries)]
    sites_filt: pd.DataFrame = \
        sites_filt.loc[:, [True] * 6 + (sites_filt.iloc[:, 6:].sum(axis=0) >= 50).tolist()]  # type: ignore

    summary = sites_filt \
        .drop(columns=["country", "continent", "region", "subregion", "geometry"]) \
        .set_index("site_id") \
        .transpose() \
        .reset_index() \
        .rename(columns={"index": f"{by}_id"})

    summary = species[[f"{by}_id", f"{by}_name"]].drop_duplicates().merge(summary, on=f"{by}_id")

    summary.insert(2, "total", summary.iloc[:, 2:].sum(axis=1))

    return summary


countries_all = [
    "Canada",
    "China",
    "Finland",
    "Germany",
    "Iran",
    "Italy",
    "Japan",
    "Lithuania",
    "Mexico",
    "Portugal",
    "Russia",
    "Slovakia",
    "South Africa",
    "Spain",
    "Switzerland",
    "Turkey",
]

countries_europe = [
    # "Canada",
    # "China",
    "Finland",
    "Germany",
    # "Iran",
    "Italy",
    # "Japan",
    "Lithuania",
    # "Mexico",
    "Portugal",
    # "Russia",
    "Slovakia",
    # "South Africa",
    "Spain",
    "Switzerland",
    # "Turkey",
]

countries_other = [
    # "Canada",
    "China",
    # "Finland",
    # "Germany",
    "Iran",
    # "Italy",
    "Japan",
    # "Lithuania",
    "Mexico",
    # "Portugal",
    "Russia",
    # "Slovakia",
    # "South Africa",
    # "Spain",
    # "Switzerland",
    # "Turkey",
]

species_by_country_europe = species_by_country[species_by_country["country"].isin(countries_europe)]
genus_by_country_europe = genus_by_country[genus_by_country["country"].isin(countries_europe)]

species_by_country_canada = species_by_country[species_by_country["country"] == "Canada"]
genus_by_country_canada = genus_by_country[genus_by_country["country"] == "Canada"]

species_by_country_other = species_by_country[species_by_country["country"].isin(countries_other)]
genus_by_country_other = genus_by_country[genus_by_country["country"].isin(countries_other)]

species_by_region = pd.concat([
    pd.DataFrame(species_by_country_europe.drop(columns=["country"]).sum(axis=0)),
    pd.DataFrame(species_by_country_canada.drop(columns=["country"]).sum(axis=0)),
    pd.DataFrame(species_by_country_other.drop(columns=["country"]).sum(axis=0)),
], axis=1).transpose()

species_by_region.insert(0, "region", ["Europe", "Canada", "Other"])

genus_by_region = pd.concat([
    pd.DataFrame(genus_by_country_europe.drop(columns=["country"]).sum(axis=0)),
    pd.DataFrame(genus_by_country_canada.drop(columns=["country"]).sum(axis=0)),
    pd.DataFrame(genus_by_country_other.drop(columns=["country"]).sum(axis=0)),
], axis=1).transpose()

genus_by_region.insert(0, "region", ["Europe", "Canada", "Other"])


# %%

def make_hist(ax: Axes, data: pd.DataFrame, cutoff: int = 10, legend_cols: int = 0, label_rot: int = 45):
    data = data.loc[:, (data != 0).any()]

    data = data.drop(
        columns=[k for k, v in data.iloc[:, 1:].sum(axis=0).items() if v <= cutoff]
    )

    ax.set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(data.shape[0], pastel_factor=0.5, rng=0)]))

    xs = np.arange(data.shape[1] - 1)
    bottom = np.zeros_like(xs)

    for _, row in data.iterrows():
        row = row.tolist()
        ax.bar(xs, row[1:], bottom=bottom, label=row[0])
        bottom += row[1:]

    ax.set_xticks(xs, labels=data.columns[1:], rotation=label_rot)
    ax.set_yscale("log")
    ax.set_ylim(10, bottom.max())
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel("total count")

    if legend_cols > 0: ax.legend(loc="upper right", ncols=legend_cols)


fig, ax = plt.subplots(2, 1, figsize=(12, 8))
make_hist(ax[0], species_by_country_europe, label_rot=90, legend_cols=3)
ax[0].set_title("By species")
make_hist(ax[1], genus_by_country_europe, label_rot=90)
ax[1].set_title("By genus")
fig.tight_layout()
fig.savefig("figures/species_genus_europe.png")
fig.show()

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
make_hist(ax[0], species_by_region, label_rot=90, legend_cols=3)
ax[0].set_title("By species")
make_hist(ax[1], genus_by_region, label_rot=90)
ax[1].axhline([1000], color="orange", linestyle="dashed")
ax[1].set_title("By genus")
fig.tight_layout()
fig.savefig("figures/species_genus_world.png")
fig.show()

# %%

labels_list = [
    "Abies",
    "Acer",
    "Betula",
    # "Castanea", # < 100
    # "Cryptomeria", # only Japan
    # "Eucalyptus", # only portugal
    "Fagus",
    # "Fraxinus", # < 1000
    # "Larix",
    "Picea",
    "Pinus",
    "Populus",
    # "Pseudotsuga", # < 1000
    "Quercus",
    "Thuja",
    # "Tilia",  # < 1000
    # "Tsuga",  # < 1000
]

labels_list = ["Background", "Other", "Dead"] + labels_list

labels = treeai \
    .get_labels() \
    .merge(species, on="label_id") \
    [["label_id", "label_name_x", "genus_name", "genus_id"]] \
    .rename(columns={"label_id": "data_label_id", "label_name_x": "data_label_name"})

labels["model_label_name"] = labels["genus_name"].apply(lambda n: n if n in labels_list else "Other")

labels.loc[labels["data_label_name"] == "Fully labeled background", "model_label_name"] = "Background"
labels.loc[labels["data_label_name"] == "Partially labeled background", "model_label_name"] = "Background"
labels.loc[labels["data_label_name"] == "Dead tree", "model_label_name"] = "Dead"

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
