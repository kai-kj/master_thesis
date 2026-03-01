# %% imports

import warnings
from pathlib import Path
import geopandas as gpd
import sklearn
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# %%

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

trees_path = Path("data/intermediate/trees_shape.gpkg")
assert trees_path.exists()

sites_summary_table_path = Path("data/intermediate/sites_shape.gpkg")

# %% load data

countries = gpd.read_file(countries_path)[["SOVEREIGNT", "geometry"]].rename(columns={"SOVEREIGNT": "country"})
print(f"found {len(countries)} country shapes")

trees = gpd.read_file(trees_path)
print(f"found {len(trees)} trees shapes")

# %% make sites table

sites_a = trees[["site_id", "geometry"]].dissolve(by="site_id").reset_index()
print(f"found {len(sites_a)} sites shapes")

# %% cleanup site shapes

sites_b = sites_a.copy()
sites_b.geometry = sites_b.geometry.concave_hull(ratio=0.4)

# %% add country information

sites_c = gpd.sjoin(sites_b, countries, how="left", predicate="within").drop(columns=["index_right"])

# %% tree species stats

species_counts = pd.crosstab(
    trees["site_id"],
    trees["species"],
).reset_index()

sites_d = sites_c.join(species_counts, how="left", on="site_id", rsuffix="_right").drop(columns=["site_id_right"])

# %%

for species in species_counts:
    print(species)

# # %% add country information
#
# trees_c = gpd.sjoin(trees_b, countries, how="left", predicate="within").drop(columns=["index_right"])
#
# # %%
#
# sites_c = sites_b[["site_id", "country", "geometry"]]
#
# # %%
#
#
# sites_summary_table_path.parent.mkdir(parents=True, exist_ok=True)
#
# sites_b.to_file(sites_summary_table_path)
