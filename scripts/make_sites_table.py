# %% imports

import warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

# %%

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

trees_path = Path("data/intermediate/trees.gpkg")
assert trees_path.exists()

sites_table_path = Path("data/intermediate/sites_by_GROUP.gpkg")

# %% load data

countries = gpd.read_file(countries_path)[["SOVEREIGNT", "geometry"]].rename(columns={"SOVEREIGNT": "country"})
print(f"found {len(countries)} country shapes")

trees = gpd.read_file(trees_path)
print(f"found {len(trees)} trees shapes")

# %% make sites table

sites_a = trees[["site_id", "geometry"]].dissolve(by="site_id").reset_index()
print(f"found {len(sites_a)} site shapes")

# %% cleanup site shapes

sites_b = sites_a.copy()
sites_b.geometry = sites_b.geometry.concave_hull(ratio=0.4)

# %% add country information

sites_c = gpd.sjoin(sites_b, countries, how="left", predicate="within").drop(columns=["index_right"])

# %% tree genus stats

for group in ["id", "species_id", "genus_id", "family_id", "order_id", "class_id", "phylum_id", "kingdom_id"]:
    species_counts = pd.crosstab(
        trees["site_id"],
        trees[group],
    ).reset_index()

    sites_d = sites_c.merge(
        species_counts, left_on="site_id", right_on="site_id", how="left"
    )

    sites_d = sites_d[["site_id", "country", *sites_d.columns[3:], "geometry"]]

    sites_table_path.parent.mkdir(parents=True, exist_ok=True)
    sites_d.to_file(sites_table_path.as_posix().replace("GROUP", group))
