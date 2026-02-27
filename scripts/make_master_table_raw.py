# %% imports

import warnings
from pathlib import Path
import geopandas as gpd

warnings.filterwarnings("ignore")

# %% setup

shapefile_path = Path("data/treeai_raw/shapes_v4/202601_FullyL_103402.shp")
assert shapefile_path.exists()

rasters_path = Path("data/treeai_raw/rasters_8bit")
assert rasters_path.exists()

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

master_table_path = Path("data/master_tables/raw.gpkg")

# %% load data

countries = gpd.read_file(countries_path)[["SOVEREIGNT", "geometry"]].rename(columns={"SOVEREIGNT": "country"})
master_table_a = gpd.read_file(shapefile_path)

# %% cleanup columns

master_table_b = master_table_a[["dataset", "speciesLat", "Tiff", "Sp_ID_int", "Biome", "geometry"]]

master_table_b = master_table_b.rename(
    columns={
        "dataset": "dataset",
        "speciesLat": "species_name",
        "Tiff": "raster",
        "Sp_ID_int": "species_id",
        "Biome": "biome",
        "geometry": "geometry",
    }
)

# %% add country information

master_table_c = gpd.sjoin(master_table_b, countries, how="left", predicate="within").drop(columns=["index_right"])

# %% raster paths

master_table_d = master_table_c
master_table_d["raster"] = master_table_d["raster"].map(lambda p: (rasters_path / p).as_posix())

for raster in master_table_d["raster"]:
    assert Path(raster).exists()

# %% reorganize columns

master_table_e = master_table_d[["species_id", "species_name", "raster", "biome", "dataset", "country", "geometry"]]

# %% save output

master_table_path.parent.mkdir(parents=True, exist_ok=True)

master_table_e.to_file(master_table_path)
master_table_e.to_csv(master_table_path.with_suffix(".csv"), index=False)
