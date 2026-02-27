# %% imports

import warnings
from pathlib import Path
import geopandas as gpd
import sklearn
import numpy as np

warnings.filterwarnings("ignore")

# %% setup

shapefile_path = Path("data/treeai_shape/shapes_v4/202601_FullyL_103402.shp")
assert shapefile_path.exists()

rasters_path = Path("data/treeai_shape/rasters_8bit")
assert rasters_path.exists()

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

trees_summary_table_path = Path("data/summary/trees_shape.gpkg")
sites_summary_table_path = Path("data/summary/sites_shape.gpkg")

# %% load data

countries = gpd.read_file(countries_path)[["SOVEREIGNT", "geometry"]].rename(columns={"SOVEREIGNT": "country"})
trees_a = gpd.read_file(shapefile_path)

trees_a.geometry = trees_a.geometry.make_valid()

# %% cleanup columns

trees_b = trees_a[["dataset", "speciesLat", "Tiff", "Sp_ID_int", "Biome", "geometry"]]

trees_b = trees_b.rename(
    columns={
        "dataset": "dataset",
        "speciesLat": "species_name",
        "Tiff": "raster",
        "Sp_ID_int": "species_id",
        "Biome": "biome",
        "geometry": "geometry",
    }
)

# %% add country information

trees_c = gpd.sjoin(trees_b, countries, how="left", predicate="within").drop(columns=["index_right"])

# %% cluster sites

centroids = np.array(
    [
        trees_c.centroid.geometry.x,
        trees_c.centroid.geometry.y,
    ], dtype=np.float32
).transpose()

centroids_clustered = sklearn.cluster \
    .DBSCAN(eps=0.0005, min_samples=5) \
    .fit(centroids) \
    .labels_

trees_d = trees_c.copy()
trees_d["site_id"] = centroids_clustered

print(f"found {trees_d['site_id'].nunique()} unique sites")

# %% sites table

sites_a = trees_d[["site_id", "geometry"]].dissolve(by="site_id").reset_index()
sites_a.geometry = sites_a.geometry.concave_hull(ratio=0.5)

# %% add country information

sites_b = gpd.sjoin(sites_a, countries, how="left", predicate="within").drop(columns=["index_right"])

# %% raster paths

trees_e = trees_d.copy()
trees_e["raster"] = trees_e["raster"].map(lambda p: (rasters_path / p).as_posix())

for raster in trees_e["raster"]:
    assert Path(raster).exists()

# %% reorganize columns

trees_f = trees_e[[
    "site_id", "species_id", "species_name", "raster", "biome", "dataset", "country", "geometry"
]]

sites_c = sites_b[["site_id", "country", "geometry"]]

# %% save output

trees_summary_table_path.parent.mkdir(parents=True, exist_ok=True)
sites_summary_table_path.parent.mkdir(parents=True, exist_ok=True)

trees_f.to_file(trees_summary_table_path)
sites_b.to_file(sites_summary_table_path)
