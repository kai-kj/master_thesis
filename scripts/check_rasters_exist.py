# %%

import warnings
from pathlib import Path
import geopandas as gpd

warnings.filterwarnings("ignore")

# %%

shapefile_path = Path("data/treeai_raw/shapes_v4/202601_FullyL_103402.shp")
assert shapefile_path.exists()

rasters_path = Path("data/treeai_raw/rasters_8bit")
assert rasters_path.exists()

# %%

shapefile = gpd.read_file(shapefile_path)

# %%

for raster in shapefile["Tiff"]:
    assert (rasters_path / raster).exists()

print("all rasters exist")
