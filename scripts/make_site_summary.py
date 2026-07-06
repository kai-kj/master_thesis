# %%

import json
from pathlib import Path

import matplotlib.pyplot as plt

# %%

dataset_names = {
    "ID": "Core data",
    "china": "China data",
    "czechia": "Czechia data",
    "iran": "Iran data",
    "mexico": "Mexico data",
    "russia": "Russia data",
    "austria": "Austria data",
    "south_africa": "South Africa data",
}

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

fig, ax = plt.subplots(2, 4, figsize=(2 * 6, 2 * 3))

for i, dataset_name in enumerate(dataset_names.keys()):
    dataset_stat_b: dict[str, int] = dataset_stats[dataset_name]["y_counts"].copy()
    dataset_stat_b.pop("background")

    ax[i // 4, i % 4].bar(dataset_stat_b.keys(), dataset_stat_b.values())  # type: ignore
    ax[i // 4, i % 4].set_xticks(range(len(dataset_stat_b)), dataset_stat_b.keys(), rotation=90)
    ax[i // 4, i % 4].spines[["top", "right"]].set_visible(False)
    ax[i // 4, i % 4].set_ylabel("occurrences (pixels)")
    ax[i // 4, i % 4].set_title(dataset_names[dataset_name])

fig.tight_layout()
fig.savefig(f"figures/site_summary.png")
fig.show()
