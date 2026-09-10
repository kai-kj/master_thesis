# %%

from pathlib import Path
import rasterio

# %%

treeai_segmentation_path = Path("/home/kai/Thesis/data/treeai_segmentation")
assert treeai_segmentation_path.is_dir()

rasters_path = treeai_segmentation_path / "rasters" / "x"
assert rasters_path.is_dir()

# %%

resolutions = []

for raster_path in rasters_path.iterdir():
    if raster_path.suffix not in [".tif", ".tiff"]:
        continue

    raster = rasterio.open(raster_path)

    print(f"{raster_path.name}")
    print(f"  size:       {raster.width} x {raster.height}")
    print(f"  bands:      {raster.count}")
    print(f"  CRS:        {raster.crs}")
    print(f"  resolution: {raster.res[0]}x{raster.res[1]} {raster.crs.linear_units}")
    print("")

    resolutions.append((raster.res[0], raster.res[1], raster.crs.linear_units))

# %%

for res in resolutions:
    print(res)
