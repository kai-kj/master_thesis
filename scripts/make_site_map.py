# %%

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# %%

data_path = Path("data/treeai/summary.gpkg")
assert data_path.exists()

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

urbans_path = Path("data/naturalearth/ne_50m_urban_areas.zip")
assert urbans_path.exists()

out_path = Path("out")

# %%

countries = gpd.read_file(countries_path)
urbans = gpd.read_file(urbans_path)

# %%

summary = gpd.read_file(data_path)
summary["name"] = summary["x"].str.split("/").str[-1].str.split(".").str[0]

# %%

# %%

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

countries.plot(ax=ax, color="lightgrey", edgecolor="lightgrey")

summary[summary["is_fully_labeled"] == False].centroid.plot(
    ax=ax, color="blue", markersize=16, label="partially labeled sites"
)

summary[summary["is_fully_labeled"] == True].centroid.plot(
    ax=ax, color="red", markersize=16, label="fully labeled sites"
)

ax.set_ylim((-60, 85))
ax.set_axis_off()
ax.legend(loc="lower left")
ax.set_title("TreeAI global sites")
fig.tight_layout()

fig.savefig(out_path / "sites_global.png")
fig.show()

# %%

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

countries.plot(ax=ax, color="lightgrey", edgecolor="darkgray")

summary[summary["is_fully_labeled"] == False].centroid.plot(
    ax=ax, color="blue", markersize=16, label="partially labeled sites"
)

summary[summary["is_fully_labeled"] == True].centroid.plot(
    ax=ax, color="red", markersize=16, label="fully labeled sites"
)

ax.set_xlim((-15, 50))
ax.set_ylim((35, 65))
ax.set_axis_off()
ax.legend(loc="upper left")
ax.set_title("TreeAI European sites")
fig.tight_layout()

fig.savefig(out_path / "sites_europe.png")
fig.show()
