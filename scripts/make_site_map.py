# %% imports

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# %% setup

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

trees_path = Path("data/intermediate/trees.gpkg")
assert trees_path.exists()

sites_path = Path("data/intermediate/sites_shape.gpkg")
assert sites_path.exists()

out_path = Path("out")

# %% load data

countries = gpd.read_file(countries_path)
trees = gpd.read_file(trees_path)
sites = gpd.read_file(sites_path)

# %% all sites

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

countries.plot(ax=ax, color="lightgrey", edgecolor="lightgrey")
sites.centroid.plot(ax=ax, color="red", markersize=16)

ax.set_ylim((-60, 85))
ax.set_axis_off()
ax.set_title("TreeAI fully labeled sites - world")
fig.tight_layout()

fig.savefig(out_path / "sites_fully_labeled_world.png")
fig.show()

# %% europe sites

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

countries.plot(ax=ax, color="lightgrey", edgecolor="darkgray")
sites.centroid.plot(ax=ax, color="red", markersize=16)

ax.set_xlim((-15, 35))
ax.set_ylim((35, 65))
ax.set_axis_off()
ax.set_title("TreeAI fully labeled sites - Europe")
fig.tight_layout()

fig.savefig(out_path / "sites_fully_labeled_europe.png")
fig.show()
