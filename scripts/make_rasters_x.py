# %% imports

import geopandas as gpd
from pathlib import Path
import numpy as np
import os

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
import cv2

# %% setup

rasters_path = Path("data/intermediate/rasters.gpkg")
assert rasters_path.exists()

chip_size: int = 512
chip_overlap: int = 64

x_output_path = Path("data/rasters/x")
preview_output_path = Path("data/rasters/preview")

# %% load data

rasters = gpd.read_file(rasters_path)

# %%

x_output_path.mkdir(parents=True, exist_ok=True)

for existing in x_output_path.glob("*.png"):
    existing.unlink()

preview_output_path.mkdir(parents=True, exist_ok=True)

for existing in preview_output_path.glob("*.png"):
    existing.unlink()

# %%

chip_stride = chip_size - 2 * chip_overlap

# rasters_tmp = rasters[rasters["path"].str.contains("48_Ortho_rgbi_Rauberberg_20240907_UAV_5cm_reproject")]

for _, raster in rasters.iterrows():
    image_name = raster['path'].split('/')[-1].split('.')[0]
    print(f"processing {image_name}")

    image = cv2.imread(raster["path"])
    assert image is not None

    assert image.shape[0] == raster["resolution_y"]
    assert image.shape[1] == raster["resolution_x"]
    assert image.shape[2] == 3

    chip_count_x = int(np.ceil(image.shape[1] / chip_stride))
    chip_count_y = int(np.ceil(image.shape[0] / chip_stride))

    padding_x = (chip_count_x * chip_stride) - image.shape[1] + chip_overlap * 2
    padding_y = (chip_count_y * chip_stride) - image.shape[0] + chip_overlap * 2

    # add black borders so that image chips cover entire image
    image_processed = cv2.copyMakeBorder(
        image,
        padding_y // 2, padding_y - padding_y // 2,
        padding_x // 2, padding_x - padding_x // 2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # remove any white/gray borders
    # gray = image_processed.mean(axis=2)
    # gray = np.stack([gray, gray, gray], axis=-1)
    # image_processed[np.all(image_processed == gray, axis=2)] = (0, 0, 0)

    # normalize
    cv2.normalize(image_processed, image_processed, 255, 0, cv2.NORM_MINMAX)

    preview_image = image_processed.copy()

    for i in np.arange(chip_count_x):
        for j in np.arange(chip_count_y):
            x0, y0 = (i * chip_stride, j * chip_stride)
            x1, y1 = (x0 + chip_size, y0 + chip_size)

            chip = image_processed[y0:y1, x0:x1]

            filename = f"{x_output_path.as_posix()}/{image_name}_{i:03d}_{j:03d}.png"

            if not np.all(chip == 0):
                print(f"  saving {filename}")
                cv2.imwrite(filename, chip)
                cv2.rectangle(preview_image, [x0 + 2, y0 + 2], [x1 - 2, y1 - 2], [0, 0, 255], 4)  # type: ignore

    filename = f"{preview_output_path.as_posix()}/{image_name}.png"
    cv2.imwrite(filename, preview_image)
