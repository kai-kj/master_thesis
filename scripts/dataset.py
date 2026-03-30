# %%

import matplotlib.pyplot as plt
import random
import pandas as pd
import torch
from pathlib import Path
import geopandas as gpd
from torchvision import tv_tensors
import torchvision.io as tv_io
import torchvision.transforms.v2 as tv_transforms

from scripts.util.seed import seed


# %%

class DatasetTreeAI(torch.utils.data.Dataset):
    def __init__(
            self,
            patches_b_path: str | Path, labels_path: str | Path,
            dist: str, split: str,
            indices: None | list[int] = None,
            transform: None | tv_transforms.Transform = None
    ):
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

        if indices is not None:
            self.patches = self.patches.iloc[indices]

        self.label_id_to_name = {r["label_id"]: r["label_name"] for _, r in pd.read_csv(labels_path).iterrows()}
        self.label_name_to_id = {v: k for k, v in self.label_id_to_name.items()}

        self.n_samples = len(self.patches)
        self.n_classes = len(self.label_id_to_name)

        count_columns = [c for c in self.patches.columns if c.endswith("_count")]
        self.class_counts = {
            kv[0].replace("_count", ""): kv[1]  # type: ignore
            for kv
            in self.patches[count_columns].sum(axis=0).items()
        }

        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Image]:
        row = self.patches.iloc[idx]

        xy = torch.cat(
            [
                tv_io.read_image(str(row["path_x"])),
                tv_io.read_image(str(row["path_y"])),
            ],
            0
        )

        if self.transform is not None:
            xy = self.transform(xy)

        x = xy[0:3, :, :].to(torch.float32) / 255.0
        y = xy[3, :, :].long()

        return tv_tensors.Image(x), tv_tensors.Image(y)


# %%

if __name__ == "__main__":
    dataset = DatasetTreeAI(
        "data/intermediate/patches_b.gpkg",
        "data/intermediate/labels.csv",
        "ID", "train",
    )

    fig, ax = plt.subplots()
    ax.bar(range(len(dataset.class_counts)), list(dataset.class_counts.values()))
    ax.set_xticks(range(len(dataset.class_counts)), labels=dataset.class_counts.keys(), rotation=90)
    ax.set_ylabel("class occurences")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.show()

# %%

if __name__ == "__main__":
    seed(42)

    basic_transform = tv_transforms.Compose([
        tv_transforms.RandomHorizontalFlip(),
        tv_transforms.RandomVerticalFlip(),
        tv_transforms.RandomRotation([0, 360]),
        tv_transforms.RandomCrop([512, 512]),
    ])

    dataset = DatasetTreeAI(
        "data/intermediate/patches_b.gpkg",
        "data/intermediate/labels.csv",
        "ID", "train",
        transform=basic_transform
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
