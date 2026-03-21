# %% imports

from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import xy as rasterio_transform_xy
import shapely
import cv2
import numpy as np
import tifffile as tiff

from scripts.check_input_outputs import check_input_outputs

# %% setup

patch_max_black: float = 0.75
patch_size: int = 1024
patch_overlap: int = 256
patch_stride = patch_size - patch_overlap

inputs: dict[str, Path] = {
    "rasters": Path("data/intermediate/rasters.csv"),
    "groups": Path("data/intermediate/groups.csv"),
}

outputs: dict[str, Path] = {
    "output_x": Path(f"data/intermediate/patches/x"),
    "output_y": Path(f"data/intermediate/patches/y"),
    "patches": Path(f"data/intermediate/patches_a.gpkg"),
}

check_input_outputs(inputs, outputs)

# %% load data

rasters = pd.read_csv(inputs["rasters"])
groups = pd.read_csv(inputs["groups"])


# %% helper function

def change_crs(shape: shapely.Geometry, src: str, dst: str) -> shapely.Geometry:
    return gpd.GeoDataFrame({"geometry": [shape]}, crs=src).to_crs(dst).geometry[0]


# %% clean output dirs

outputs["output_x"].mkdir(parents=True, exist_ok=True)
outputs["output_y"].mkdir(parents=True, exist_ok=True)

for f in outputs["output_x"].glob("*"): f.unlink()
for f in outputs["output_y"].glob("*"): f.unlink()

# %%

patches = []

for _, row in rasters.iterrows():
    print(f"processing {row['site_id']}")

    image_x = tiff.imread(row["path_x"])
    image_y = tiff.imread(row["path_y"])
    image_y = image_y.reshape(image_y.shape[0], image_y.shape[1], 1)

    assert image_x.shape[0:2] == image_y.shape[0:2]
    assert image_x.shape[2] == 3
    assert image_y.shape[2] == 1

    patch_count_x = int(np.ceil(image_x.shape[1] / patch_stride))
    patch_count_y = int(np.ceil(image_x.shape[0] / patch_stride))

    padding_x = (patch_count_x * patch_stride) - image_x.shape[1] + patch_overlap
    padding_y = (patch_count_y * patch_stride) - image_x.shape[0] + patch_overlap

    image_x = cv2.copyMakeBorder(
        image_x,
        padding_y // 2, padding_y - padding_y // 2,
        padding_x // 2, padding_x - padding_x // 2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    image_y = cv2.copyMakeBorder(
        image_y,
        padding_y // 2, padding_y - padding_y // 2,
        padding_x // 2, padding_x - padding_x // 2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    n_rasters = 0

    for i in np.arange(patch_count_x):
        for j in np.arange(patch_count_y):
            x0, y0 = (i * patch_stride, j * patch_stride)
            x1, y1 = (x0 + patch_size, y0 + patch_size)

            patch_x = image_x[y0:y1, x0:x1].copy()
            patch_y = image_y[y0:y1, x0:x1].copy()

            # some basic normalization
            cv2.normalize(patch_x, patch_x, 255, 0, cv2.NORM_MINMAX)

            # skip images with a lot of black
            if np.all(patch_x == [0, 0, 0], axis=-1).sum() / (
                    patch_x.shape[0] * patch_x.shape[1]) > patch_max_black:
                continue

            patch_path_x = f"{outputs['output_x'].as_posix()}/{row['site_id']}_{i:03d}_{j:03d}.png"
            patch_path_y = f"{outputs['output_y'].as_posix()}/{row['site_id']}_{i:03d}_{j:03d}.png"

            cv2.imwrite(patch_path_x, patch_x)
            cv2.imwrite(patch_path_y, patch_y)

            with rasterio.open(row["path_x"]) as raster_x:
                bounds = rasterio_transform_xy(
                    raster_x.transform,
                    [y0 - padding_y // 2, y1 - padding_y // 2],
                    [x0 - padding_x // 2, x1 - padding_x // 2]
                )
                bounds = shapely.geometry.box(min(bounds[0]), min(bounds[1]), max(bounds[0]), max(bounds[1]))
                bounds = change_crs(bounds, raster_x.crs, "EPSG:4326")  # type: ignore

                patches.append((row["site_id"], patch_path_x, patch_path_y, bounds))
                n_rasters += 1

    print(f"  created {n_rasters} rasters")

patches = gpd.GeoDataFrame(patches, columns=["site_id", "path_x", "path_y", "geometry"], crs="EPSG:4326")

# %%

patches.to_file(outputs["patches"])
