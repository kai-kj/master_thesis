# %%

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import json

from scripts.dataset import DatasetTreeAI

# %%

dataset_names = {
    "ID": "Core data",
    "OOD_china": "China data",
    "OOD_czechia": "Czechia data",
    "OOD_iran": "Iran data",
    "OOD_mexico": "China data",
    "OOD_russia": "Russia data",
    "OOD_south_africa": "South Africa data",
}

# %%

dataset_stats = {}

for dataset_name in dataset_names.keys():
    print(f"calculating {dataset_name} stats")
    dataset = DatasetTreeAI(dataset_name, "train", 0, dataset_name == "OOD_south_africa")
    dataset_stats[dataset_name] = dataset.calc_stats()[2]

# %%

for dataset_name in dataset_names.keys():
    dataset_stat: dict[str, int] = dataset_stats[dataset_name].copy()
    dataset_stat.pop("background")

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(dataset_stat.keys(), dataset_stat.values())  # type: ignore
    ax.set_xticks(range(len(dataset_stat)), dataset_stat.keys(), rotation=20)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("occurrences (pixels)")

    fig.tight_layout()
    fig.savefig(f"figures/site_summary_{dataset_name}.png")
    fig.show()
