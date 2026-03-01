# %% imports

import warnings
from pathlib import Path
import geopandas as gpd
import sklearn
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# %% setup

shapefile_path = Path("data/treeai_shape/shapes_v4/202601_FullyL_103402.shp")
assert shapefile_path.exists()

rasters_path = Path("data/treeai_shape/rasters_8bit")
assert rasters_path.exists()

species_path = Path("data/intermediate/species.csv")
assert species_path.exists()

trees_summary_table_path = Path("data/intermediate/trees_shape.gpkg")

# %% load data

trees_a = gpd.read_file(shapefile_path)
species = pd.read_csv(species_path)

# %% cleanup columns

trees_b = trees_a[["dataset", "speciesLat", "Tiff", "Biome", "geometry"]]

trees_b = trees_b.rename(
    columns={
        "dataset": "dataset",
        "speciesLat": "species",
        "Tiff": "raster",
        "Biome": "biome",
        "geometry": "geometry",
    }
)

print(f"found {len(trees_b)} tree shapes")
print(f"found {trees_b['species'].nunique()} unique species")

# %% fix geometry

print(f"found {sum(~trees_b.geometry.is_valid)} invalid geometries")

trees_c = trees_b.copy()
trees_c.geometry = trees_c.geometry.make_valid()

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

# %% raster paths

trees_e = trees_d.copy()
trees_e["raster"] = trees_e["raster"].map(lambda p: (rasters_path / p).as_posix())

for raster in trees_e["raster"]:
    assert Path(raster).exists()

print("all raster paths are valid")

# %% reorganize columns

trees_f = trees_e[[
    "site_id", "species", "raster", "biome", "dataset", "geometry"
]]

# %% save output

trees_summary_table_path.parent.mkdir(parents=True, exist_ok=True)
trees_f.to_file(trees_summary_table_path)

print(f"saved tree shapes table to {trees_summary_table_path}")
