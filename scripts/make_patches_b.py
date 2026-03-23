# %% imports

from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

from scripts.check_input_outputs import check_input_outputs

# %% setup

inputs: dict[str, Path] = {
    "patches_a": Path("data/intermediate/patches_a.gpkg"),
    "groups": Path("data/intermediate/groups.csv"),
    "labels": Path("data/intermediate/labels.csv"),
}

outputs: dict[str, Path] = {
    "patches_b": Path("data/intermediate/patches_b.gpkg"),
}

check_input_outputs(inputs, outputs)

# %%

groups = pd.read_csv(inputs["groups"])
patches_a = gpd.read_file(inputs["patches_a"])
labels = pd.read_csv(inputs["labels"])

# %%

target_ratio = 0.6

patches_b = patches_a.merge(groups, on="site_id", how="left")

for group_id in patches_b["group_id"].unique():
    print(group_id)

    group = patches_b[patches_b["group_id"] == group_id]
    sites = group[["site_id"]].value_counts().items()
    sites = sorted([(s[0], c) for s, c in sites], key=lambda x: x[1], reverse=True)

    train_sites = []
    test_sites = []

    train_count = 0
    test_count = 0

    for site, count in sites:
        if train_count == 0 or train_count / (train_count + test_count) < target_ratio:
            train_sites.append(site)
            train_count += count
        else:
            test_sites.append(site)
            test_count += count

    print(f"  train ratio: {train_count / (train_count + test_count)}")

    patches_b.loc[patches_b["site_id"].isin(train_sites), "split"] = "train"
    patches_b.loc[patches_b.site_id.isin(test_sites), "split"] = "test"

# %%

total_counts = patches_b.groupby("group_id")["split"].count()
train_counts = patches_b[patches_b["split"] == "train"].groupby("group_id")["split"].count()
train_ratios = train_counts / total_counts

actual_ratio = len(patches_b[patches_b["split"] == "train"]) / len(patches_b)

fig, ax = plt.subplots()

ax.bar(range(len(train_ratios)), train_ratios, label="train", alpha=0.6)
ax.bar(range(len(train_ratios)), 1 - train_ratios, bottom=train_ratios, label="test", alpha=0.6)
ax.hlines(target_ratio, -1, len(train_ratios), linestyles="--", color="r", label="target ratio")
ax.hlines(actual_ratio, -1, len(train_ratios), linestyles=":", color="b", label="actual ratio")

ax.set_xticks(range(len(train_ratios)), labels=train_ratios.index, rotation=90)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right")

fig.tight_layout()
fig.show()

# %%

label_counts = []

for path_y in patches_b["path_y"]:
    image: np.ndarray = cv2.imread(path_y, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]  # type: ignore
    hist = cv2.calcHist([image], [0], None, [len(labels)], [0, len(labels)]).astype(np.int64)

    assert np.sum(hist) == image.shape[0] * image.shape[1]
    label_counts.append([path_y] + list(hist.flatten()))

label_count_col_names = [f"{l}_count" for l in labels["label_name"]]
label_counts = pd.DataFrame(label_counts, columns=["path_y"] + label_count_col_names)

# %%

patches_c = patches_b \
    .merge(label_counts, on="path_y", how="left") \
    [["site_id", "group_id", "dist", "split", "path_x", "path_y"] + label_count_col_names + ["geometry"]]

# %%

patches_c.to_file(outputs["patches_b"])
