# %%

from pathlib import Path
import pandas as pd
import geopandas as gpd

# %%

data_path = Path("data/treeai/summary_a.csv")
assert data_path.exists()

world_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert world_path.exists()

out_path = Path("data/treeai/summary_b.csv")

# %%

world = gpd.read_file(world_path)

# %%

summary = pd.read_csv(data_path)

summary = gpd.GeoDataFrame(
    summary[["label_type", "x", "y", "lat", "lon"]],
    geometry=gpd.points_from_xy(summary["lon"], summary["lat"]),
    crs="EPSG:4326",
)

summary = gpd.sjoin(
    summary,
    world[["SOVEREIGNT", "geometry"]],
    how="left",
    predicate="within"
)

summary = summary.drop(columns=["geometry", "index_right"])
summary = summary.rename(columns={"SOVEREIGNT": "country"})

# %%

print("Countries:")

for country in summary["country"].unique():
    n_part = len(summary[(summary["country"] == country) & (summary["label_type"] == "part")])
    n_full = len(summary[(summary["country"] == country) & (summary["label_type"] == "full")])

    print(f"- {country}")
    print(f"  - part: {n_part}")
    print(f"  - full: {n_full}")

# %%

out_path.parent.mkdir(parents=True, exist_ok=True)
summary = summary.to_csv(out_path, index=False)
