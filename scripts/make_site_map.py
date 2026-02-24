# %%

from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# %%

data_path = Path("data/treeai/summary_a.csv")
assert data_path.exists()

world_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert world_path.exists()

out_path = Path("out")

# %%

world = gpd.read_file(world_path)

# %%

summary = pd.read_csv(data_path)
summary["name"] = summary["x"].str.split("/").str[-1].str.split(".").str[0]
summary = summary[["name", "label_type", "lat", "lon"]]

summary = gpd.GeoDataFrame(
    summary[["name", "label_type"]],
    geometry=gpd.points_from_xy(summary["lon"], summary["lat"]),
    crs="EPSG:4326",
)

# %%

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

world.plot(ax=ax, color="lightgrey", edgecolor="lightgrey")
summary[summary["label_type"] == "part"].plot(ax=ax, color="blue", markersize=16, label="partially labeled sites")
summary[summary["label_type"] == "full"].plot(ax=ax, color="red", markersize=16, label="fully labeled sites")

ax.set_ylim((-60, 85))
ax.set_axis_off()
ax.legend(loc="lower left")
ax.set_title("TreeAI global sites")
fig.tight_layout()

fig.savefig(out_path / "sites_global.png")
fig.show()

# %%

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

world.plot(ax=ax, color="lightgrey", edgecolor="darkgrey")
summary[summary["label_type"] == "part"].plot(ax=ax, color="blue", markersize=16, label="partially labeled sites")
summary[summary["label_type"] == "full"].plot(ax=ax, color="red", markersize=16, label="fully labeled sites")

ax.set_xlim((-15, 50))
ax.set_ylim((35, 65))
ax.set_axis_off()
ax.legend(loc="upper left")
ax.set_title("TreeAI European sites")
fig.tight_layout()

fig.savefig(out_path / "sites_europe.png")
fig.show()
