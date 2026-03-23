# %%

import matplotlib.pyplot as plt
import random
import pandas as pd
import torch
from pathlib import Path
import geopandas as gpd
from torchvision import tv_tensors
import torchvision.io as tv_io


# %%

class DatasetTreeAI(torch.utils.data.Dataset):
    def __init__(self, patches_b_path: str | Path, labels_path: str | Path, dist: str, split: str):
        super(DatasetTreeAI, self).__init__()

        patches_b_path = Path(patches_b_path)
        labels_path = Path(labels_path)

        assert patches_b_path.exists()
        assert labels_path.exists()
        assert dist in ["ID", "OOD"]
        assert split in ["train", "test"]

        self.dist = dist
        self.split = split

        self.patches = gpd.read_file(patches_b_path).reset_index(drop=True)
        self.patches = self.patches[self.patches["dist"] == self.dist]
        self.patches = self.patches[self.patches["split"] == self.split]

        self.label_id_to_name = {r["label_id"]: r["label_name"] for _, r in pd.read_csv(labels_path).iterrows()}
        self.label_name_to_id = {v: k for k, v in self.label_id_to_name.items()}

        self.n_samples = len(self.patches)
        self.n_classes = len(self.label_id_to_name)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Image]:
        row = self.patches.iloc[idx]

        x = tv_io.read_image(str(row["path_x"])).to(torch.float32) / 255.0
        y = tv_io.read_image(str(row["path_y"])).long()

        return tv_tensors.Image(x), tv_tensors.Image(y)


# %%

if __name__ == "__main__":
    random.seed(42)

    dataset = DatasetTreeAI(
        "data/intermediate/patches_b.gpkg",
        "data/intermediate/labels.csv",
        "ID", "train",
    )

    n_samples = 5
    fig, ax = plt.subplots(2, n_samples, figsize=(n_samples * 4, 8))

    for i in range(n_samples):
        index = random.randint(0, len(dataset) - 1)
        x, y = dataset[index]

        x = x.numpy().transpose(1, 2, 0)
        y = y.numpy().transpose(1, 2, 0)

        ax[0, i].imshow(x, cmap="gray")
        ax[0, i].axis("off")

        vmin = min(dataset.label_id_to_name.keys())

        ax[1, i].imshow(
            y,
            vmin=min(dataset.label_id_to_name.keys()), vmax=max(dataset.label_id_to_name.keys()),
            cmap="viridis"
        )
        ax[1, i].axis("off")

    fig.tight_layout()
    fig.show()
