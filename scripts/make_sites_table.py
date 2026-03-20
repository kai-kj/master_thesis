# %% imports

import warnings
from pathlib import Path
import geopandas as gpd

warnings.filterwarnings("ignore")

# %%

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

trees_a_path = Path("data/intermediate/trees_a.gpkg")
assert trees_a_path.exists()

sites_path = Path("data/intermediate/sites_a.gpkg")

# %% load data

countries = gpd.read_file(countries_path)[["NAME_LONG", "CONTINENT", "REGION_UN", "SUBREGION", "geometry"]]
countries = countries.rename(
    columns={"NAME_LONG": "country", "CONTINENT": "continent", "REGION_UN": "region", "SUBREGION": "subregion"}
)

print(f"found {len(countries)} country shapes")

trees = gpd.read_file(trees_a_path)
print(f"found {len(trees)} trees shapes")

# %% make sites table

sites_a = trees[["site_id", "geometry"]].dissolve(by="site_id").reset_index()
print(f"found {len(sites_a)} site shapes")

# %% cleanup site shapes

sites_b = sites_a.copy()
sites_b.geometry = sites_b.geometry.concave_hull(ratio=1)

# %% add country information

sites_c = gpd.sjoin(sites_b, countries, how="left", predicate="within").drop(columns=["index_right"])

# %%

sites_c = sites_c[["site_id", "country", "continent", "region", "subregion", "geometry"]]

sites_path.parent.mkdir(parents=True, exist_ok=True)
sites_c.to_file(sites_path)
