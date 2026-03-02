# %% imports

import warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
import shapely

# %% setup

trees_table_path = Path("data/intermediate/trees.gpkg")
assert trees_table_path.exists()

rasters_table_path = Path("data/intermediate/rasters.gpkg")

# %% load data

trees = gpd.read_file(trees_table_path)

# %%

raster_paths = trees["raster"].unique().tolist()

rasters_a = []

for raster_path in raster_paths:
    with rasterio.open(raster_path) as raster:
        image_size = (raster.width, raster.height)
        pixel_size = (raster.transform[0], -raster.transform[4])
        geometry = shapely.geometry.box(*raster.bounds)
        crs = str(raster.crs)

        assert pixel_size[0] == pixel_size[1]
        pixel_size = round(pixel_size[0], 4)

        row = gpd.GeoDataFrame(
            [[raster_path, *image_size, pixel_size, crs, geometry]],
            columns=["path", "resolution_x", "resolution_y", "pixel_size", "orig_crs", "geometry"],
            crs=raster.crs,
        ).to_crs(trees.crs)  # type: ignore

        rasters_a.append(row)

rasters_a = gpd.GeoDataFrame(pd.concat(rasters_a))

print(f"found {len(rasters_a)} rasters")

# %%

rasters_table_path.parent.mkdir(parents=True, exist_ok=True)
rasters_a.to_file(rasters_table_path)
