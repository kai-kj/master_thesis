# %% imports

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
import numpy as np
from cycler import cycler
from distinctipy import get_colors, get_hex

# %% setup

sites_path = Path("data/intermediate/sites_a.gpkg")
assert sites_path.exists()

trees_a_path = Path("data/intermediate/trees_a.gpkg")
assert trees_a_path.exists()

species_path = Path("data/intermediate/species.csv")
assert species_path.exists()

labels_path = Path("data/intermediate/labels.csv")
labels_map_path = Path("data/intermediate/labels_map.csv")
out_path = Path("out")

# %% load data

sites = gpd.read_file(sites_path)
trees = gpd.read_file(trees_a_path)
species = pd.read_csv(species_path)

# %%

sites_species = sites.merge(
    pd.crosstab(trees["site_id"], trees["species_id"], dropna=False).reset_index(),
    on="site_id"
)

sites_genera = sites.merge(
    pd.crosstab(trees["site_id"], trees["genus_id"], dropna=False).reset_index(),
    on="site_id"
)

black_forest_sites = [f"SITE{s:03X}" for s in range(0x02b + 1)]
other_sites = ["SITE042", "SITE043", "SITE044", "SITE045"]

sites_species.loc[sites_species["site_id"].isin(black_forest_sites + other_sites), "country"] = "Germany S"
sites_genera.loc[sites_genera["site_id"].isin(black_forest_sites + other_sites), "country"] = "Germany S"

sites_species.loc[sites_species["country"] == "Germany", "country"] = "Germany N"
sites_genera.loc[sites_genera["country"] == "Germany", "country"] = "Germany N"


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
    # "Germany",
    "Germany N",
    "Germany S",
    "Iran",
    "Italy",
    "Japan",
    "Lithuania",
    "Mexico",
    "Portugal",
    "Russian Federation",
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
    # "Germany",
    "Germany N",
    "Germany S",
    # "Iran",
    "Italy",
    # "Japan",
    "Lithuania",
    # "Mexico",
    "Portugal",
    # "Russian Federation",
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
    # "Germany N",
    # "Germany S",
    "Iran",
    # "Italy",
    "Japan",
    # "Lithuania",
    "Mexico",
    # "Portugal",
    "Russian Federation",
    # "Slovakia",
    # "South Africa",
    # "Spain",
    # "Switzerland",
    # "Turkey",
]

species_country_all, genus_country_all = {}, {}

for c in countries_all:
    species_country_all[c] = make_site_summary(sites_species, "species", [c])
    genus_country_all[c] = make_site_summary(sites_genera, "genus", [c])

species_country_europe = {k: v for k, v in species_country_all.items() if k in countries_europe}
genus_country_europe = {k: v for k, v in genus_country_all.items() if k in countries_europe}

species_europe = make_site_summary(sites_species, "species", countries_europe)
genus_europe = make_site_summary(sites_genera, "genus", countries_europe)

species_canada = make_site_summary(sites_species, "species", ["Canada"])
genus_canada = make_site_summary(sites_genera, "genus", ["Canada"])

species_other = make_site_summary(sites_species, "species", countries_other)
genus_other = make_site_summary(sites_genera, "genus", countries_other)


# %%

def make_combined_summary(sources: list[pd.DataFrame], names: list[str], by: str):
    assert len(sources) == len(names)

    combined = pd.DataFrame(pd.concat(sources)[f"{by}_id"].unique(), columns=[f"{by}_id"])

    for s, n in zip(sources, names):
        combined = combined \
            .merge(s[[f"{by}_id", "total"]], on=f"{by}_id", how="left") \
            .rename(columns={"total": n})

    combined = species[[f"{by}_id", f"{by}_name"]] \
        .drop_duplicates() \
        .merge(combined, on=f"{by}_id")

    combined.iloc[:, :2] = combined.iloc[:, :2].fillna("(other)")
    combined.iloc[:, 2:] = combined.iloc[:, 2:].fillna(0)

    combined = combined.sort_values(f"{by}_name").reset_index(drop=True)

    return combined


species_combined_all = make_combined_summary(
    list(species_country_all.values()),
    list(species_country_all.keys()),
    "species"
)

genus_combined_all = make_combined_summary(
    list(genus_country_all.values()),
    list(genus_country_all.keys()),
    "genus"
)

species_combined_europe = make_combined_summary(
    list(species_country_europe.values()),
    list(species_country_europe.keys()),
    "species"
)

genus_combined_europe = make_combined_summary(
    list(genus_country_europe.values()),
    list(genus_country_europe.keys()),
    "genus"
)

species_combined_group = make_combined_summary(
    [species_europe, species_canada, species_other],
    ["europe", "canada", "other"],
    "species",
)

genus_combined_group = make_combined_summary(
    [genus_europe, genus_canada, genus_other],
    ["europe", "canada", "other"],
    "genus",
)


# %%

def make_hist(ax: Axes, data: pd.DataFrame, by: str, legend_cols: int = 0, label_rot: int = 45):
    ax.set_prop_cycle(cycler(color=[get_hex(c) for c in get_colors(len(data), pastel_factor=0.5, rng=0)]))

    xs = np.arange(len(data))
    bottom = np.zeros_like(xs)

    for c in list(data.columns)[2:]:
        ax.bar(xs, data[c], bottom=bottom, label=c)
        bottom += data[c]

    ax.set_xticks(xs, labels=list(data[f"{by}_name"]), rotation=label_rot)
    ax.set_yscale("log")
    ax.set_ylim(10, bottom.max())
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel("total count")

    if legend_cols > 0: ax.legend(loc="upper right", ncols=legend_cols)


fig, ax = plt.subplots(2, 1, figsize=(12, 8))
make_hist(ax[0], species_combined_europe, "species", legend_cols=3)
ax[0].set_title("By species")
make_hist(ax[1], genus_combined_europe, "genus")
ax[1].set_title("By genus")
fig.tight_layout()
fig.savefig(out_path / "species_genus_by_region_europe.png")
fig.show()

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
make_hist(ax[0], species_combined_group, "species", legend_cols=3, label_rot=90)
ax[0].set_title("By species")
make_hist(ax[1], genus_combined_group, "genus")
ax[1].hlines([100], -1, len(genus_combined_group), color="k", linestyle="dashed")
ax[1].set_title("By genus")
fig.tight_layout()
fig.savefig(out_path / "species_genus_by_region_global.png")
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
    "Fraxinus",
    "Larix",
    "Picea",
    "Pinus",
    "Populus",
    "Pseudotsuga",
    "Quercus",
    "Thuja",
    "Tilia",
    "Tsuga",
]
#
# labels_list += ["Other", "Dead", "Background"]
#
# print(f"labels ({len(labels_list)}: {labels_list}")

# %%

labels_map = species[["id", "genus_name"]] \
    .copy() \
    .rename(columns={"id": "tree_id", "genus_name": "label"})

labels_map.loc[~labels_map["label"].isin(labels_list), "label"] = "Other"

dead_tree_id = list(species.loc[species["name"] == "Dead tree", "id"])[0]
labels_map.loc[labels_map["tree_id"] == dead_tree_id, "label"] = "Dead"

labels_map = pd.concat([labels_map, pd.DataFrame([{"tree_id": None, "label": "Background"}])], ignore_index=True)

label_to_id = {v: i + 3 for i, v in enumerate(labels_list)}
label_to_id["Background"] = 0
label_to_id["Other"] = 1
label_to_id["Dead"] = 2

labels_map["label_id"] = labels_map["label"].map(lambda v: label_to_id[v])
labels_map = labels_map[["tree_id", "label_id", "label"]]

# %%

labels = pd.DataFrame(labels_map["label"].unique(), columns=["label"]) \
    .merge(species[["genus_name", "genus_id"]].drop_duplicates(), left_on="label", right_on="genus_name", how="left") \
    .drop(columns="genus_name") \
    .rename(columns={"label": "name", "genus_id": "description"})

labels["description"] = labels["description"].map(
    lambda v: f"https://catalogueoflife.org/data/taxon/{v}" if not pd.isna(v) else ""
)

labels.loc[labels["name"] == "Dead", "description"] = "Dead tree of any genus"
labels.loc[labels["name"] == "Other", "description"] = "Other tree genera"
labels.loc[labels["name"] == "Background", "description"] = "Background class"

labels["id"] = labels["name"].map(lambda v: label_to_id[v])
labels = labels[["id", "name", "description"]].sort_values("id").reset_index(drop=True)

assert labels["id"].nunique() == labels["name"].nunique()

print(f"{len(labels)} labels:")

for _, r in labels.iterrows():
    print(f"- {r['name']}: {r['description']}")

# %%

labels_path.parent.mkdir(parents=True, exist_ok=True)
labels.to_csv(labels_path, index=False)

labels_map_path.parent.mkdir(parents=True, exist_ok=True)
labels_map.to_csv(labels_map_path, index=False)
