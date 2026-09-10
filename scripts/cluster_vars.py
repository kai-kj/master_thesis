# %%

import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation
from scripts.colors import palette

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# %%

treeai = TreeAISegmentation("/home/kai/Thesis/data/treeai_segmentation/")

# %%

patches_clim = treeai.get_fusion_patches("512", "clim").drop(columns=["patch_name", "geometry"])
patches_soil = treeai.get_fusion_patches("512", "soil").drop(columns=["patch_name", "geometry"])

patches = treeai.get_patches_ext("512") \
    .merge(gpd.read_file("data/sites_split.gpkg"), on="site_id", how="left") \
    [["subpatch_name", "dist"]] \
    .merge(patches_clim, on="subpatch_name", how="left") \
    .merge(patches_soil, on="subpatch_name", how="left")

# %%

pc = [[0, 2], [1, 3]]

patches_x = patches[list(patches.columns)[2:]]
patches_x_scaled = StandardScaler().fit_transform(patches_x)

pca = PCA()
patches_x_pca = pca.fit_transform(patches_x_scaled)
explained = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

for i, dist in enumerate(patches["dist"].unique()):
    if dist == "OOD":
        continue

    ax[0].scatter(
        patches_x_pca[patches["dist"] == dist, pc[0][0]],
        patches_x_pca[patches["dist"] == dist, pc[0][1]],
        label=dist.replace("ID", "core").replace("OOD_", "").replace("_", " ").title(),
        marker="o" if dist != "ID" else "^",
        color=palette(len(patches["dist"].unique()), i, pastel_factor=0.25),
        s=20
    )

    ax[1].scatter(
        patches_x_pca[patches["dist"] == dist, pc[1][0]],
        patches_x_pca[patches["dist"] == dist, pc[1][1]],
        label=dist.replace("ID", "core").replace("OOD_", "").replace("_", " ").title(),
        marker="o" if dist != "ID" else "^",
        color=palette(len(patches["dist"].unique()), i, pastel_factor=0.25),
        s=20
    )

ax[0].axhline(0, color="lightgray", linestyle=":", zorder=0)
ax[0].axvline(0, color="lightgray", linestyle=":", zorder=0)

ax[1].axhline(0, color="lightgray", linestyle=":", zorder=0)
ax[1].axvline(0, color="lightgray", linestyle=":", zorder=0)

ax[0].set_xlabel(f"PC {pc[0][0] + 1} ({explained[pc[0][0]]:.1f}%)")
ax[0].set_ylabel(f"PC {pc[0][1] + 1} ({explained[pc[0][1]]:.1f}%)")
ax[0].spines[["top", "right"]].set_visible(False)

ax[1].set_xlabel(f"PC {pc[1][0] + 1} ({explained[pc[1][0]]:.1f}%)")
ax[1].set_ylabel(f"PC {pc[1][1] + 1} ({explained[pc[1][1]]:.1f}%)")
ax[1].legend(loc="lower right", frameon=False, ncol=1)
ax[1].spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("figures/site_env_pca.png")
fig.show()
