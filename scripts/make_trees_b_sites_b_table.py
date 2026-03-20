# %%

import geopandas as gpd
import pandas as pd
from pathlib import Path

# %%

trees_a_path = Path("data/intermediate/trees_a.gpkg")
assert trees_a_path.exists()

sites_a_path = Path("data/intermediate/sites_a.gpkg")
assert sites_a_path.exists()

groups_path = Path("data/intermediate/groups.csv")
assert groups_path.exists()

labels_map_path = Path("data/intermediate/labels_map.csv")
assert labels_map_path.exists()

labels_path = Path("data/intermediate/labels.csv")
assert labels_path.exists()

trees_b_path = Path("data/intermediate/trees_b.gpkg")
sites_b_path = Path("data/intermediate/sites_b.gpkg")

# %%

trees_a = gpd.read_file(trees_a_path)
sites_a = gpd.read_file(sites_a_path)
groups = pd.read_csv(groups_path)
labels_map = pd.read_csv(labels_map_path)
labels = pd.read_csv(labels_path)

# %%

trees_b = trees_a \
    .merge(groups, left_on="site_id", right_on="site_id", how="left") \
    .merge(labels_map, left_on="id", right_on="tree_id", how="left") \
    [["id", "site_id", "group_id", "distribution", "label_id", "label", "raster", "geometry"]] \
    .rename(columns={"id": "tree_id", "label": "label_name"})

trees_b = trees_b[~trees_b["group_id"].isna()]

# %%

sites_b = sites_a \
    .merge(groups, left_on="site_id", right_on="site_id", how="left") \
    [["site_id", "group_id", "distribution", "geometry"]]

# %%

trees_b_path.parent.mkdir(parents=True, exist_ok=True)
trees_b.to_file(trees_b_path)

sites_b_path.parent.mkdir(parents=True, exist_ok=True)
sites_b.to_file(sites_b_path)
