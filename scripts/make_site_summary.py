# %%

from pathlib import Path
import matplotlib.pyplot as plt
import json

from scripts.dataset import DatasetTreeAI

# %%

dataset_stats_path = Path("data/dataset_stats_detailed.json")

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
    print(f"loading {dataset_name}")
    dataset_train = DatasetTreeAI(dataset_name, "train", 0, dataset_name == "OOD_south_africa")
    dataset_test = DatasetTreeAI(dataset_name, "test", 0, dataset_name == "OOD_south_africa")

    print(f"calculating {dataset_name} stats")
    dataset_stat_a = dataset_train.calc_stats()

    dataset_stats[dataset_name] = {
        "x_mean": [float(v) for v in dataset_stat_a[0]],
        "x_std": [float(v) for v in dataset_stat_a[1]],
        "label_counts": dataset_stat_a[2],
        "train_size": len(dataset_train),
        "test_size": len(dataset_test),
    }

dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
dataset_stats_path.write_text(json.dumps(dataset_stats, indent=4))

# %%

for dataset_name in dataset_names.keys():
    dataset_stat_b: dict[str, int] = dataset_stats[dataset_name]["label_counts"].copy()
    dataset_stat_b.pop("background")

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(dataset_stat_b.keys(), dataset_stat_b.values())  # type: ignore
    ax.set_xticks(range(len(dataset_stat_b)), dataset_stat_b.keys(), rotation=20)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("occurrences (pixels)")

    fig.tight_layout()
    fig.savefig(f"figures/site_summary_{dataset_name}.png")
    fig.show()

# %%

print("train sizes:")

for dataset_name in dataset_names.keys():
    print(f"- {dataset_name}: {dataset_stats[dataset_name]["train_size"]}")

# FT sizes: 5, 10, 50, 100
