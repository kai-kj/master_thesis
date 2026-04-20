# %%

import matplotlib.pyplot as plt
import random
import pandas as pd
import torch
from pathlib import Path
from torchvision import tv_tensors
import torchvision.io as tv_io
import torchvision.transforms.v2 as tv_transforms
import numpy as np
import json
import geopandas as gpd

from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation


# %%

class DatasetTreeAI(torch.utils.data.Dataset):
    TREEAI_SEGMENTATION_PATH = Path("/home/kai/Thesis/data/treeai_segmentation/")
    SITE_SPLIT_PATH = Path("data/sites_split.gpkg")
    LABELS_MAP_PATH = Path("data/labels_map.csv")

    def __init__(
            self,
            id_or_ood: str, train_or_test: str, split_n: int,
            indices: None | list[int] = None,
            xy_transform: None | tv_transforms.Transform = None,
            x_transform: None | tv_transforms.Transform = None,
            y_transform: None | tv_transforms.Transform = None,
    ):
        super(DatasetTreeAI, self).__init__()

        assert self.TREEAI_SEGMENTATION_PATH.exists()
        assert self.SITE_SPLIT_PATH.exists()
        assert self.LABELS_MAP_PATH.exists()
        assert id_or_ood in ["ID", "OOD"]
        assert train_or_test in ["train", "test"]
        assert split_n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self._treeai = TreeAISegmentation(self.TREEAI_SEGMENTATION_PATH)

        self._patches = self._treeai.get_patches_512x512_ext() \
            .merge(gpd.read_file(self.SITE_SPLIT_PATH)[["site_id", "dist"]], on="site_id") \
            .drop(columns=["geometry"])

        self._patches = self._patches[self._patches["dist"] == id_or_ood]
        self._patches = self._patches[self._patches[f"split_{split_n}"] == train_or_test]

        self._patches = self._patches[["crop_x0", "crop_x1", "crop_y0", "crop_y1", "path_x", "path_y"]]

        if indices is not None:
            self._patches = self._patches.iloc[indices]

        self._xy_transform = xy_transform
        self._x_transform = x_transform
        self._y_transform = y_transform

        labels_map = pd.read_csv(self.LABELS_MAP_PATH)

        self._labels_lut = torch.arange(labels_map["data_label_id"].max() + 1)  # type: ignore

        for _, r in labels_map.iterrows():
            self._labels_lut[r["data_label_id"]] = r["model_label_id"]

        self.label_id_to_name = {int(r["model_label_id"]): str(r["model_label_name"]) for _, r in labels_map.iterrows()}
        self.label_name_to_id = {str(r["model_label_name"]): int(r["model_label_id"]) for _, r in labels_map.iterrows()}

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Image]:
        x_tensor, y_tensor = self.get_raw(idx)

        if self._x_transform is not None: x_tensor = self._x_transform(x_tensor)
        if self._y_transform is not None: y_tensor = self._y_transform(y_tensor)

        return x_tensor, y_tensor  # type: ignore

    def get_raw(self, idx) -> tuple[tv_tensors.Image, tv_tensors.Image]:
        row = self._patches.iloc[idx]

        xy_tensor = torch.cat([tv_io.read_image(row["path_x"]), tv_io.read_image(row["path_y"])], 0)  # type: ignore
        xy_tensor = xy_tensor[:, row["crop_y0"]:row["crop_y1"], row["crop_x0"]:row["crop_x1"]]  # type: ignore

        if self._xy_transform is not None: xy_tensor = self._xy_transform(xy_tensor)

        x_tensor = tv_tensors.Image(xy_tensor[0:3, :, :].to(torch.float32) / 255.0)
        y_tensor = self._labels_lut[tv_tensors.Image(xy_tensor[3, :, :].long())]

        return x_tensor, y_tensor  # type: ignore

    def calc_stats(self) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        n_pixels = 0
        x_sum = np.zeros(3)
        x_sq_sum = np.zeros(3)
        y_counts = np.zeros(len(dataset.label_id_to_name), dtype=np.int64)

        for i in range(len(dataset)):
            x, y = dataset[i]
            x, y = x.numpy(), y.numpy()

            n_pixels += x.shape[1] * x.shape[2]
            x_sum += x.sum(axis=(1, 2))
            x_sq_sum += (x ** 2).sum(axis=(1, 2))
            y_counts += np.bincount(y.ravel(), minlength=len(dataset.label_id_to_name))

        x_mean = x_sum / n_pixels
        x_std = np.sqrt(x_sq_sum / n_pixels - (x_sum / n_pixels) ** 2)
        label_counts = {dataset.label_id_to_name[i]: int(y_counts[i]) for i in range(len(y_counts))}

        return x_mean, x_std, label_counts  # type: ignore


# %%

if __name__ == "__main__":
    dataset = DatasetTreeAI("ID", "train", 0)
    dataset_stats = dataset.calc_stats()

    fig, ax = plt.subplots()
    ax.bar(range(len(dataset_stats[2])), list(dataset_stats[2].values()))
    ax.set_xticks(range(len(dataset_stats[2])), labels=dataset_stats[2].keys(), rotation=90)
    ax.set_ylabel("class occurences")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.show()

    dataset_stats = {
        "x_mean": list(dataset_stats[0]),
        "x_std": list(dataset_stats[1]),
        "y_counts": dataset_stats[2],
    }

    Path("data/dataset_stats.json").write_text(json.dumps(dataset_stats, indent=4))

# %%

if __name__ == "__main__":
    dataset = DatasetTreeAI(
        "ID", "train", 0,
        xy_transform=tv_transforms.Compose([
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.RandomVerticalFlip(),
            tv_transforms.RandomRotation([0, 90]),
        ]),
        x_transform=tv_transforms.Normalize([0.24, 0.38, 0.29], [0.19, 0.27, 0.24]),  # from dataset_stats.json
    )

    n_samples = 5
    fig, ax = plt.subplots(2, n_samples, figsize=(n_samples * 4, 8))

    for i in range(n_samples):
        index = random.randint(0, len(dataset) - 1)
        x, y = dataset[index]

        x = x.numpy().transpose(1, 2, 0)
        y = y.numpy().transpose(1, 2, 0)

        ax[0, i].imshow((x - x.min()) / (x.max() - x.min()))
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
