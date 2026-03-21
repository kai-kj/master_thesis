# %%

import geopandas as gpd
import pandas as pd
from pathlib import Path

from scripts.check_input_outputs import check_input_outputs

# %%

inputs: dict[str, Path] = {
    "trees_a": Path("data/intermediate/trees_a.gpkg"),
    "sites_a": Path("data/intermediate/sites_a.gpkg"),
    "groups": Path("data/intermediate/groups.csv"),
    "labels_map": Path("data/intermediate/labels_map.csv"),
    "labels": Path("data/intermediate/labels.csv"),
}

outputs: dict[str, Path] = {
    "trees_b": Path("data/intermediate/trees_b.gpkg"),
    "sites_b": Path("data/intermediate/sites_b.gpkg"),
}

check_input_outputs(inputs, outputs)

# %%

trees_a = gpd.read_file(inputs["trees_a"])
sites_a = gpd.read_file(inputs["sites_a"])
groups = pd.read_csv(inputs["groups"])
labels_map = pd.read_csv(inputs["labels_map"])
labels = pd.read_csv(inputs["labels"])

# %%

trees_b = trees_a \
    .merge(groups, on="site_id", how="left") \
    .merge(labels_map, on="tree_id", how="left") \
    [["tree_id", "site_id", "group_id", "dist", "label_id", "label_name", "raster", "geometry"]]

trees_b = trees_b[~trees_b["group_id"].isna()]

# %%

sites_b = sites_a \
    .merge(groups, on="site_id", how="left") \
    [["site_id", "group_id", "dist", "geometry"]]

# %%

trees_b.to_file(outputs["trees_b"])
sites_b.to_file(outputs["sites_b"])
