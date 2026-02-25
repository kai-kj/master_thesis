# %%

from pathlib import Path
import pandas as pd
import re
import utm
from PIL import Image
import shapely
import geopandas as gpd

# %%

paths_table_path = Path("data/treeai/paths.csv")
assert paths_table_path.exists()

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

urbans_path = Path("data/naturalearth/ne_50m_urban_areas.zip")
assert urbans_path.exists()

out_path = Path("data/treeai/summary.gpkg")

# %%

countries = gpd.read_file(countries_path)[["SOVEREIGNT", "geometry"]].rename(columns={"SOVEREIGNT": "country"})
urbans = gpd.read_file(urbans_path)[["geometry"]]

# %%

paths = pd.read_csv(paths_table_path)

summary = []

for _, row in paths.iterrows():
    is_fully_labeled = row["is_fully_labeled"]
    x = row["x"]
    y = row["y"]
    world = Path(row["world"]).read_text().split("\n")
    meta = Path(row["meta"]).read_text()

    world_a = float(world[0])
    world_d = float(world[1])
    world_b = float(world[2])
    world_e = float(world[3])
    world_c = float(world[4])
    world_f = float(world[5])

    assert world_d == 0
    assert world_b == 0

    utm_zone = re.search(r"UTM zone (\w+)", meta).group(1)  # type: ignore

    utm_northern = utm_zone[-1] == "N"
    utm_zone = int(utm_zone[:-1])

    image = Image.open(row["x"])
    size_x, size_y = image.size

    x_0, y_0 = (world_c - 0.5 * world_a, world_f - 0.5 * world_e)
    x_1, y_1 = (world_c + (size_x + 0.5) * world_a, world_f + (size_y + 0.5) * world_e)

    lat_0, lon_0 = utm.to_latlon(x_0, y_0, utm_zone, northern=utm_northern, strict=False)
    lat_1, lon_1 = utm.to_latlon(x_1, y_1, utm_zone, northern=utm_northern, strict=False)

    geometry = shapely.box(lon_0, lat_0, lon_1, lat_1)

    summary.append((is_fully_labeled, x, y, geometry))

summary = gpd.GeoDataFrame(summary, columns=["is_fully_labeled", "x", "y", "geometry"], crs="EPSG:4326")

# %%

summary_2 = gpd.sjoin(summary, countries, how="left", predicate="within").drop(columns=["index_right"])

# %%

summary_3 = summary_2.copy()
summary_3["is_urban"] = ~gpd.sjoin(summary_2, urbans, how="left", predicate="within")["index_right"].isna()

# %%

summary_4 = summary_3[["is_fully_labeled", "is_urban", "country", "x", "y", "geometry"]]

# %%

out_path.parent.mkdir(parents=True, exist_ok=True)
summary_4.to_file(out_path, index=False)
