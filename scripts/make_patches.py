# %% imports

from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.plot import reshape_as_image
from rasterio.plot import reshape_as_raster
from rasterio.transform import xy as rasterio_transform_xy
from rasterio.enums import Resampling
import shapely
import cv2
import numpy as np

# %% setup

patch_max_black: float = 0.8
patch_size: int = 512
patch_overlap: int = 128
patch_stride = patch_size - 2 * patch_overlap

trees_path = Path("data/intermediate/trees.gpkg")
assert trees_path.exists()

sites_path = Path("data/intermediate/sites.gpkg")
assert sites_path.exists()

name = f"patches_{patch_size}x{patch_size}_{patch_overlap}"

merged_outputs_path = Path("data/intermediate/rasters_merged")
x_output_path = Path(f"data/intermediate/patches/{patch_size}x{patch_size}_{patch_overlap}/x")
y_output_path = Path(f"data/intermediate/patches/{patch_size}x{patch_size}_{patch_overlap}/y")
patches_path = Path(f"data/intermediate/patches_{patch_size}x{patch_size}_{patch_overlap}.gpkg")

# %% load data

trees = gpd.read_file(trees_path)
sites = gpd.read_file(sites_path)


# %% helper function

def change_crs(shape: shapely.Geometry, src: str, dst: str) -> shapely.Geometry:
    return gpd.GeoDataFrame({"geometry": [shape]}, crs=src).to_crs(dst).geometry[0]


# %% clean output dirs

merged_outputs_path.mkdir(parents=True, exist_ok=True)
x_output_path.mkdir(parents=True, exist_ok=True)
y_output_path.mkdir(parents=True, exist_ok=True)

for f in merged_outputs_path.glob("*"): f.unlink()
for f in x_output_path.glob("*"): f.unlink()
for f in y_output_path.glob("*"): f.unlink()

# %%

# hard_sites = ["SITE054", "SITE048", "SITE05A", "SITE06C"]
# hard_sites = ["SITE057", "SITE058", "SITE059"]
# hard_sites = sites[sites["site_id"].isin(hard_sites)]

# %% process patches

patches = []

for _, site in sites.iterrows():
    name = site['site_id']
    merged_path = merged_outputs_path / f"{name}.tif"

    print(f"processing {name}")

    all_sources = []
    raster_crs = []

    for s in list(trees[trees["site_id"] == site["site_id"]]["raster"].unique()):
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

    site_bounds = change_crs(shapely.box(*site["geometry"].bounds), sites.crs, raster_crs)  # type: ignore

    rasterio_merge(
        sources, bounds=site_bounds.bounds,  # type: ignore
        use_highest_res=True, resampling=Resampling.lanczos, method="max", dst_path=merged_path
    )

    with (rasterio.open(merged_path, "r+") as raster):
        print(f"  size: {raster.width}x{raster.height}")
        image = reshape_as_image(raster.read())[:, :, [2, 1, 0]]

        # remove any white/gray borders
        h0 = cv2.calcHist([image], [0], None, [256], (0, 256)).reshape(-1)
        h1 = cv2.calcHist([image], [1], None, [256], (0, 256)).reshape(-1)
        h2 = cv2.calcHist([image], [2], None, [256], (0, 256)).reshape(-1)

        bg = np.array([h0.argsort()[-1], h1.argsort()[-1], h2.argsort()[-1]])
        bg = np.array([h0.argsort()[-2], h1.argsort()[-2], h2.argsort()[-2]]) if np.all(bg == [0, 0, 0]) else bg

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = np.all(image == bg, axis=-1).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(np.bool)

        image[mask] = [0, 0, 0]

        # save at this stage so that qgis gets nicer images to show
        raster.write(reshape_as_raster(image[:, :, [2, 1, 0]]))

        # for qgis
        with open(f"{merged_path}.aux.xml", "w") as aux_file:
            aux_file.write("""
                <PAMDataset>
                  <PAMRasterBand band="1"><NoDataValue>-1.00000000000000E+00</NoDataValue></PAMRasterBand>
                  <PAMRasterBand band="2"><NoDataValue>-1.00000000000000E+00</NoDataValue></PAMRasterBand>
                  <PAMRasterBand band="3"><NoDataValue>-1.00000000000000E+00</NoDataValue></PAMRasterBand>
                </PAMDataset>
            """)

        # pad images to get maximum chip cover
        patch_count_x = int(np.ceil(image.shape[1] / patch_stride))
        patch_count_y = int(np.ceil(image.shape[0] / patch_stride))

        padding_x = (patch_count_x * patch_stride) - image.shape[1] + patch_overlap * 2
        padding_y = (patch_count_y * patch_stride) - image.shape[0] + patch_overlap * 2

        image = cv2.copyMakeBorder(
            image,
            padding_y // 2, padding_y - padding_y // 2,
            padding_x // 2, padding_x - padding_x // 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        n_rasters = 0

        for i in np.arange(patch_count_x):
            for j in np.arange(patch_count_y):
                x0, y0 = (i * patch_stride, j * patch_stride)
                x1, y1 = (x0 + patch_size, y0 + patch_size)

                chip = image[y0:y1, x0:x1].copy()

                # some basic normalization
                cv2.normalize(chip, chip, 255, 0, cv2.NORM_MINMAX)

                # skip images with a lot of black
                if np.all(chip == [0, 0, 0], axis=-1).sum() / (chip.shape[0] * chip.shape[1]) > patch_max_black:
                    continue

                patch_path = f"{x_output_path.as_posix()}/{name}_{i:03d}_{j:03d}.png"
                cv2.imwrite(patch_path, chip)

                bounds = rasterio_transform_xy(
                    raster.transform,
                    [y0 - padding_y // 2, y1 - padding_y // 2],
                    [x0 - padding_x // 2, x1 - padding_x // 2]
                )
                bounds = shapely.geometry.box(min(bounds[0]), min(bounds[1]), max(bounds[0]), max(bounds[1]))
                bounds = change_crs(bounds, raster.crs, sites.crs)  # type: ignore

                patches.append((patch_path, name, i, j, bounds))

                n_rasters += 1

        print(f"  created {n_rasters} rasters")

patches = gpd.GeoDataFrame(patches, columns=["path", "site_id", "row", "col", "geometry"], crs=sites.crs)

# %%

patches.to_file(patches_path)
