# %% imports

import warnings
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# %% setup

countries_path = Path("data/naturalearth/ne_50m_admin_0_countries.zip")
assert countries_path.exists()

master_table_path = Path("data/master_tables/raw.gpkg")
assert master_table_path.exists()

out_path = Path("out")

# %% load data

countries = gpd.read_file(countries_path)
master_table = gpd.read_file(master_table_path)

# %% all sites

fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

countries.plot(ax=ax, color="lightgrey", edgecolor="lightgrey")
master_table.centroid.plot(ax=ax, color="red", markersize=16)

ax.set_ylim((-60, 85))
ax.set_axis_off()
ax.set_title("TreeAI fully labeled sites - world")
fig.tight_layout()

fig.savefig(out_path / "sites_fully_labeled_world.png")
fig.show()

# %% europe sites

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

countries.plot(ax=ax, color="lightgrey", edgecolor="darkgray")
master_table.centroid.plot(ax=ax, color="red", markersize=16)

ax.set_xlim((-15, 35))
ax.set_ylim((35, 65))
ax.set_axis_off()
ax.set_title("TreeAI fully labeled sites - Europe")
fig.tight_layout()

fig.savefig(out_path / "sites_fully_labeled_europe.png")
fig.show()
