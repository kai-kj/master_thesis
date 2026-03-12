# %% imports

import warnings
from pathlib import Path
import geopandas as gpd
import sklearn
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# %% setup

shapefile_path = Path("data/treeai/shapes_v4/202601_FullyL_103402.shp")
assert shapefile_path.exists()

rasters_path = Path("data/treeai/rasters_8bit")
assert rasters_path.exists()

species_path = Path("data/intermediate/species.csv")
assert species_path.exists()

trees_table_path = Path("data/intermediate/trees.gpkg")

# %% load data

trees_a = gpd.read_file(shapefile_path)
species = pd.read_csv(species_path)

# %% cleanup columns

trees_b = trees_a[["dataset", "speciesLat", "Tiff", "geometry"]]

levels = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]

trees_b = trees_b.merge(
    species[["name", "id"] + [ll for l in levels for ll in [f"{l}_id", f"{l}_name"]]],
    left_on="speciesLat",
    right_on="name",
    how="left",
)

trees_b = trees_b.rename(columns={"Tiff": "raster", "geometry": "geometry", })

trees_b = trees_b[["id"] + [ll for l in levels for ll in [f"{l}_id", f"{l}_name"]] + ["raster", "geometry"]]

print(f"found {len(trees_b)} tree shapes")
print(f"  {trees_b['id'].nunique()} unique labels")
print(f"  {trees_b['species_id'].nunique()} unique (valid) species")
print(f"  {trees_b['genus_id'].nunique()} unique (valid) genera")
print(f"  {trees_b['family_id'].nunique()} unique (valid) families")
print(f"  {trees_b['order_id'].nunique()} unique (valid) orders")
print(f"  {trees_b['class_id'].nunique()} unique (valid) classes")
print(f"  {trees_b['kingdom_id'].nunique()} unique (valid) kingdoms")

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
trees_d.insert(0, "site_id", centroids_clustered)
trees_d["site_id"] = trees_d["site_id"].map(lambda i: f"SITE{i:03X}")

print(f"found {trees_d['site_id'].nunique()} unique sites")

# %% raster paths

trees_e = trees_d.copy()
trees_e["raster"] = trees_e["raster"].map(lambda p: (rasters_path / p).as_posix())

for raster in trees_e["raster"]:
    assert Path(raster).exists()

print("all raster paths are valid")

# %% save output

trees_table_path.parent.mkdir(parents=True, exist_ok=True)
trees_e.to_file(trees_table_path)

print(f"saved tree shapes table to {trees_table_path}")
