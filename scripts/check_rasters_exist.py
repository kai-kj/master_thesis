# %%

import warnings
from pathlib import Path
import geopandas as gpd

from scripts.check_input_outputs import check_input_outputs

warnings.filterwarnings("ignore")

# %%

inputs: dict[str, Path] = {
    "shapefile": Path("data/treeai/shapes_v4/202601_FullyL_103402.shp"),
    "rasters": Path("data/treeai/rasters_8bit"),
}

check_input_outputs(inputs, None)

# %%

shapefile = gpd.read_file(inputs["shapefile"])

# %%

for raster in shapefile["Tiff"]:
    assert (inputs["rasters"] / raster).exists()

print("all rasters exist")
