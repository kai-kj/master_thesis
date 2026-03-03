# %% imports

from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rasterio_merge
from rasterio.plot import reshape_as_image
from rasterio.plot import reshape_as_raster
from rasterio.transform import xy as rasterio_transform_xy
import shapely
import cv2
import numpy as np

# %% setup

trees_path = Path("data/intermediate/trees.gpkg")
assert trees_path.exists()

sites_path = Path("data/intermediate/sites.gpkg")
assert sites_path.exists()

merged_outputs_path = Path("data/rasters/merged")
x_output_path = Path("data/rasters/x")

rasters_path = Path("data/intermediate/rasters.gpkg")

chip_max_black: float = 0.75
chip_size: int = 512
chip_overlap: int = 64
chip_stride = chip_size - 2 * chip_overlap

# %% load data

trees = gpd.read_file(trees_path)
sites = gpd.read_file(sites_path)


# %% helper function

def change_crs(shape: shapely.Geometry, src: str, dst: str) -> shapely.Geometry:
    return gpd.GeoDataFrame({"geometry": [shape]}, crs=src).to_crs(dst).geometry[0]


# %% clean output dirs

merged_outputs_path.mkdir(parents=True, exist_ok=True)
x_output_path.mkdir(parents=True, exist_ok=True)

for f in merged_outputs_path.glob("*"): f.unlink()
for f in x_output_path.glob("*"): f.unlink()

# %% process rasters

rasters = []

for _, site in sites.iterrows():
    name = site['site_id']
    merged_path = merged_outputs_path / f"{name}.tif"

    print(f"processing {name}")

    rasterio_merge(
        trees[trees["site_id"] == site["site_id"]]["raster"].unique(),
        method="max", dst_path=merged_path
    )

    with (rasterio.open(merged_path, "r+") as raster):
        image = reshape_as_image(raster.read())[:, :, [2, 1, 0]]

        # remove any white/gray borders
        hist_0 = cv2.calcHist([image], [0], None, [256], (0, 256))
        hist_1 = cv2.calcHist([image], [1], None, [256], (0, 256))
        hist_2 = cv2.calcHist([image], [2], None, [256], (0, 256))

        bg = np.array([hist_0.argmax(), hist_1.argmax(), hist_2.argmax()])

        if np.all(bg == [0, 0, 0]):
            hist_0[bg[0]], hist_1[bg[0]], hist_2[bg[0]] = 0, 0, 0
            bg = np.array([hist_0.argmax(), hist_1.argmax(), hist_2.argmax()])

        image[image == bg] = 0

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
        chip_count_x = int(np.ceil(image.shape[1] / chip_stride))
        chip_count_y = int(np.ceil(image.shape[0] / chip_stride))

        padding_x = (chip_count_x * chip_stride) - image.shape[1] + chip_overlap * 2
        padding_y = (chip_count_y * chip_stride) - image.shape[0] + chip_overlap * 2

        image = cv2.copyMakeBorder(
            image,
            padding_y // 2, padding_y - padding_y // 2,
            padding_x // 2, padding_x - padding_x // 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        for i in np.arange(chip_count_x):
            for j in np.arange(chip_count_y):
                x0, y0 = (i * chip_stride, j * chip_stride)
                x1, y1 = (x0 + chip_size, y0 + chip_size)

                chip = image[y0:y1, x0:x1].copy()

                # some basic normalization
                cv2.normalize(chip, chip, 255, 0, cv2.NORM_MINMAX)

                # skip images with a lot of black
                if np.all(chip == [0, 0, 0], axis=-1).sum() / (chip.shape[0] * chip.shape[1]) > chip_max_black:
                    continue

                chip_path = f"{x_output_path.as_posix()}/{name}_{i:03d}_{j:03d}.png"
                cv2.imwrite(chip_path, chip)

                bounds = rasterio_transform_xy(
                    raster.transform,
                    [y0 - padding_y // 2, y1 - padding_y // 2],
                    [x0 - padding_x // 2, x1 - padding_x // 2]
                )
                bounds = shapely.geometry.box(min(bounds[0]), min(bounds[1]), max(bounds[0]), max(bounds[1]))
                bounds = change_crs(bounds, raster.crs, sites.crs)  # type: ignore

                rasters.append((chip_path, name, i, j, bounds))

rasters = gpd.GeoDataFrame(rasters, columns=["path", "site_id", "row", "col", "geometry"], crs=sites.crs)

# %%

rasters.to_file(rasters_path, index=False)
