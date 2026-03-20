# %% imports

from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.plot import reshape_as_image
from rasterio.plot import reshape_as_raster
from rasterio.features import rasterize as rasterio_rasterize
from rasterio.enums import Resampling
import shapely
import cv2
import numpy as np

# %% setup

trees_path = Path("data/intermediate/trees_b.gpkg")
assert trees_path.exists()

sites_path = Path("data/intermediate/sites_b.gpkg")
assert sites_path.exists()

x_output_path = Path("data/intermediate/rasters/x")
y_output_path = Path("data/intermediate/rasters/y")
rasters_path = Path("data/intermediate/rasters.csv")

# %% load data

trees = gpd.read_file(trees_path)
sites = gpd.read_file(sites_path)


# %% helper function

def change_crs(shape: shapely.Geometry, src: str, dst: str) -> shapely.Geometry:
    return gpd.GeoDataFrame({"geometry": [shape]}, crs=src).to_crs(dst).geometry[0]


# %% clean output dirs

x_output_path.mkdir(parents=True, exist_ok=True)
y_output_path.mkdir(parents=True, exist_ok=True)

for f in x_output_path.glob("*"): f.unlink()
for f in y_output_path.glob("*"): f.unlink()

# %%

rasters = []

for site_id in trees["site_id"].unique().tolist():
    # for site_id in ["SITE042", "SITE055", "SITE056"]:
    path_x = x_output_path / f"{site_id}.tif"
    path_y = y_output_path / f"{site_id}.tif"

    print(f"processing {site_id}")

    all_sources = []
    raster_crs = []

    for s in list(trees[trees["site_id"] == site_id]["raster"].unique()):
        with rasterio.open(s) as r:
            bounds = np.array(r.bounds).round().astype(np.uint32)
            all_sources.append((s, *bounds))
            raster_crs.append(str(r.crs))

    raster_crs = list(set(raster_crs))
    assert len(raster_crs) == 1
    raster_crs = raster_crs[0]

    # remove fully overlapping sources
    sources = pd.DataFrame(all_sources, columns=["path", "b0", "b1", "b2", "b3"]) \
        .sort_values(by="path", ascending=False) \
        .groupby(["b0", "b1", "b2", "b3"]) \
        .aggregate("first")["path"] \
        .tolist()

    print(f"  sources:")
    for s in all_sources:
        print(f"  + {s[0]}" if s[0] in sources else f"  - {s[0]}")

    site_bounds = change_crs(
        shapely.box(*sites.loc[sites["site_id"] == site_id, "geometry"].item().bounds),  # type: ignore
        sites.crs, raster_crs,  # type: ignore
    )

    rasterio_merge(
        sources, bounds=site_bounds.bounds,  # type: ignore
        use_highest_res=True, resampling=Resampling.lanczos, method="max", dst_path=path_x
    )

    with rasterio.open(path_x, "r+") as raster_x:
        image_x = reshape_as_image(raster_x.read())[:, :, [2, 1, 0]]

        print(f"  size: {image_x.shape[0]}x{image_x.shape[1]}x{image_x.shape[2]}")

        # remove any white/gray borders
        h0 = cv2.calcHist([image_x], [0], None, [256], (0, 256)).reshape(-1)
        h1 = cv2.calcHist([image_x], [1], None, [256], (0, 256)).reshape(-1)
        h2 = cv2.calcHist([image_x], [2], None, [256], (0, 256)).reshape(-1)

        bg = np.array([h0.argsort()[-1], h1.argsort()[-1], h2.argsort()[-1]])
        bg = np.array([h0.argsort()[-2], h1.argsort()[-2], h2.argsort()[-2]]) if np.all(bg == [0, 0, 0]) else bg

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = np.all(image_x == bg, axis=-1).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(np.bool)

        image_x[mask] = [0, 0, 0]
        image_x[image_x == [255, 255, 255]] = 0

        image_x = image_x.astype(np.float32) / 255.0
        # p_low, p_high = np.percentile(image_x, (2.5, 97.5))
        p_low, p_high = np.percentile(image_x, (1, 99))
        image_x = np.clip(image_x, p_low, p_high)
        image_x = (image_x - p_low) / (p_high - p_low)
        image_x = (image_x * 255).astype(np.uint8)

        raster_x.write(reshape_as_raster(image_x[:, :, [2, 1, 0]]))

        # for qgis
        with open(f"{path_x}.aux.xml", "w") as aux_file:
            aux_file.write("""
                <PAMDataset>
                  <PAMRasterBand band="1"><NoDataValue>-1.00000000000000E+00</NoDataValue></PAMRasterBand>
                  <PAMRasterBand band="2"><NoDataValue>-1.00000000000000E+00</NoDataValue></PAMRasterBand>
                  <PAMRasterBand band="3"><NoDataValue>-1.00000000000000E+00</NoDataValue></PAMRasterBand>
                </PAMDataset>
            """)

        profile_y = raster_x.profile.copy()
        profile_y.update(count=1, dtype="uint8")

        shapes = [
            (change_crs(r["geometry"], trees.crs, raster_crs), r["label_id"])  # type: ignore
            for _, r
            in trees[trees["site_id"] == site_id].iterrows()
        ]

        rasterio_rasterize(shapes=shapes, fill=0, nodata=0, dtype="uint8", dst_path=path_y, dst_kwds=profile_y)

        with rasterio.open(path_y, "r") as raster_y:
            image_y = reshape_as_image(raster_y.read())
            assert len(np.unique(image_y)) \
                   == trees.loc[trees["site_id"] == site_id, "label_id"].nunique() + 1  # type: ignore

        rasters.append((site_id, path_x.as_posix(), path_y.as_posix()))

rasters = pd.DataFrame(rasters, columns=["site_id", "path_x", "path_y"])

# %%

rasters_path.parent.mkdir(parents=True, exist_ok=True)
rasters.to_csv(rasters_path, index=False)
