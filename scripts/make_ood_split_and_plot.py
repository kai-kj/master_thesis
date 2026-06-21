# # %% imports
#
# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# import numpy as np
# from cycler import cycler
# from distinctipy import get_colors, get_hex
#
# from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation
#
# # %%
#
# sites_split_path = Path("data/sites_split.gpkg")
# assert sites_split_path.is_file()
#
# countries_path = Path("/home/kai/Thesis/data/naturalearth/ne_10m_admin_0_countries.zip")
# assert countries_path.is_file()
#
# # %%
#
# sites_split_part = gpd.read_file(sites_split_path)
# countries = gpd.read_file(countries_path)
#
# # %%
#
# ood_country_province = [
#     ("Iran", "North Khorasan"),
#     ("Mexico", "Veracruz"),
#     ("Czechia", "Plzeňský"),
#     ("South Africa", "Limpopo"),
#     ("China", "Anhui"),
#     ("Russia", "Primor'ye"),
# ]
#
# sites_split = sites_split_part.copy()
#
# for country, province in ood_country_province:
#     dist_name = f"OOD_{country.lower().replace(' ', '_')}"
#     sites_split.loc[(sites_split["country"] == country) & (sites_split["province"] == province), "dist"] = dist_name
#
# # %%
#
# fig, ax = plt.subplots(figsize=(16, 8))
#
# countries.to_crs("EPSG:3857").plot(ax=ax, color="lightgray")
#
# sites_split[sites_split["dist"] == "OOD"] \
#     .to_crs("EPSG:3857") \
#     .centroid \
#     .plot(ax=ax, color="gray", marker="^", markersize=100, label="unused")
#
# dists = sorted(list(set(sites_split["dist"].unique()) - {"OOD"}))
# colors = [get_hex(c) for c in get_colors(len(dists), pastel_factor=0, rng=0)]
#
# for dist, color in zip(dists, colors):
#     sites_split[sites_split["dist"] == dist] \
#         .to_crs("EPSG:3857") \
#         .centroid \
#         .plot(ax=ax, color=color, markersize=100, label=dist.replace("_", " "))
#
# ax.axis("off")
# ax.set_xlim((-20000000, 20000000))
# ax.set_ylim((-20000000 * 0.4, 20000000 * 0.6))
# ax.legend(loc="lower left")
# fig.tight_layout()
# fig.savefig("figures/id_ood_full_map.png")
# fig.show()
#
# # %%
