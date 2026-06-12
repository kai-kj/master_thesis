# %%

import random
from pathlib import Path
import json
from typing import Literal

import albumentations as A
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import geopandas as gpd
import cv2

from dependencies.treeai_segmentation.treeai_segmentation import TreeAISegmentation


# %%

class DatasetTreeAI(torch.utils.data.Dataset):
    TREEAI_SEGMENTATION_PATH = Path("/home/kai/Thesis/data/treeai_segmentation/")
    SITE_SPLIT_PATH = Path("data/sites_split.gpkg")
    LABELS_MAP_PATH = Path("data/labels_map.csv")
    LABELS_MAP_PATH_AFRICA = Path("data/labels_map_africa.csv")

    def __init__(
            self,
            dist: str, train_or_test: str, split_n: int,
            use_africa_labels: bool = False,
            indices: None | list[int] = None,
            augmentations: A.Compose | None = None,
    ):
        super(DatasetTreeAI, self).__init__()

        assert self.TREEAI_SEGMENTATION_PATH.exists()
        assert self.SITE_SPLIT_PATH.exists()
        assert self.LABELS_MAP_PATH.exists()
        assert self.LABELS_MAP_PATH_AFRICA.exists()
        assert dist.startswith("ID") or dist.startswith("OOD")
        assert train_or_test in ["train", "test"]
        assert split_n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # for when calling .split()
        self._config = {
            "dist": dist, "train_or_test": train_or_test, "split_n": split_n, "use_africa_labels": use_africa_labels
        }

        self._treeai = TreeAISegmentation(self.TREEAI_SEGMENTATION_PATH)

        self._patches = self._treeai.get_patches_512x512_ext() \
            .merge(gpd.read_file(self.SITE_SPLIT_PATH)[["site_id", "dist"]], on="site_id") \
            .drop(columns=["geometry"])

        self._patches = self._patches[self._patches["dist"] == dist]
        self._patches = self._patches[self._patches[f"split_{split_n}"] == train_or_test]

        self._patches = self._patches[["site_id", "crop_x0", "crop_x1", "crop_y0", "crop_y1", "path_x", "path_y"]]

        if indices is not None:
            self._patches = self._patches.iloc[indices]

        self._augmentations = augmentations

        labels_map = pd.read_csv(self.LABELS_MAP_PATH) \
            if not use_africa_labels \
            else pd.read_csv(self.LABELS_MAP_PATH_AFRICA)

        self._labels_lut = torch.arange(labels_map["data_label_id"].max() + 1)  # type: ignore

        for _, r in labels_map.iterrows():
            self._labels_lut[r["data_label_id"]] = r["model_label_id"]

        self.label_id_to_name = {int(r["model_label_id"]): str(r["model_label_name"]) for _, r in labels_map.iterrows()}
        self.label_name_to_id = {str(r["model_label_name"]): int(r["model_label_id"]) for _, r in labels_map.iterrows()}

    def sample(self, n: int, replace: bool = False, random_state: int | None = None):
        self._patches = self._patches.sample(n, replace=replace, random_state=random_state)

    def split(
            self,
            test_size: float,
            train_augmentations: A.Compose | None = None,
            test_augmentations: A.Compose | None = None,
            random_state: int | None = None
    ) -> tuple["DatasetTreeAI", "DatasetTreeAI"]:
        train_idx, test_idx = sklearn.model_selection.train_test_split(
            range(len(self)), test_size=test_size, random_state=random_state
        )

        train_dataset = DatasetTreeAI(indices=train_idx, **self._config)
        if train_augmentations is not None: train_dataset._augmentations = train_augmentations

        test_dataset = DatasetTreeAI(indices=test_idx, **self._config)
        if test_augmentations is not None: test_dataset._augmentations = test_augmentations

        return train_dataset, test_dataset

    def loader(
            self, batch_size: int,
            num_workers: int = 4,
            prefetch_factor: int = 2,
            others: list["DatasetTreeAI"] = [],
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self if len(others) == 0 else torch.utils.data.ConcatDataset([self, *others]),
            batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=True
        )

    def calc_stats(self) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        n_pixels = 0
        x_sum = np.zeros(3)
        x_sq_sum = np.zeros(3)
        y_counts = np.zeros(len(self.label_id_to_name), dtype=np.int64)

        for i in range(len(self)):
            x, y = self[i]
            x, y = x.numpy(), y.numpy()

            n_pixels += x.shape[1] * x.shape[2]
            x_sum += x.sum(axis=(1, 2))
            x_sq_sum += (x ** 2).sum(axis=(1, 2))
            y_counts += np.bincount(y.ravel(), minlength=len(self.label_id_to_name))

        x_mean = x_sum / n_pixels
        x_std = np.sqrt(x_sq_sum / n_pixels - (x_sum / n_pixels) ** 2)
        label_counts = {self.label_id_to_name[i]: int(y_counts[i]) for i in range(len(y_counts))}

        return x_mean, x_std, label_counts  # type: ignore

    def get_xy(self, idx, apply_augmentations: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._patches.iloc[idx]

        x_array: np.ndarray = cv2.cvtColor(cv2.imread(row["path_x"]), cv2.COLOR_BGR2RGB)  # type: ignore
        y_array: np.ndarray = cv2.imread(row["path_y"], cv2.IMREAD_GRAYSCALE)  # type: ignore

        x_array = x_array[row["crop_y0"]:row["crop_y1"], row["crop_x0"]:row["crop_x1"], :]
        y_array = y_array[row["crop_y0"]:row["crop_y1"], row["crop_x0"]:row["crop_x1"]]

        x_array = x_array.astype(np.float32) / 255.0

        if apply_augmentations and (self._augmentations is not None):
            augmented = self._augmentations(image=x_array, mask=y_array)
            x_array = augmented["image"]
            y_array = augmented["mask"]

        x_tensor = torch.Tensor(x_array).permute(2, 0, 1)
        y_tensor = self._labels_lut[torch.Tensor(y_array).long()]

        return x_tensor, y_tensor

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_xy(idx, apply_augmentations=True)


def make_base_train_augmentations(
        x_mean: tuple[float, float, float],
        x_std: tuple[float, float, float]
) -> A.Compose:
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-90.0, 90.0), p=1.0),

        # Sensor
        # A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.5),
        # A.HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-5, 5), val_shift_limit=(-5, 5), p=0.5),
        # A.GaussianBlur(blur_limit=(3, 3), p=0.125),

        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10), p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), p=0.125),

        # Norm
        A.Normalize(mean=x_mean, std=x_std, max_pixel_value=1.0, p=1.0),
    ])


def make_base_test_augmentations(
        x_mean: tuple[float, float, float],
        x_std: tuple[float, float, float]
) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=x_mean, std=x_std, max_pixel_value=1.0, p=1.0),
    ])


# %%

if __name__ == "__main__":
    dataset = DatasetTreeAI("ID", "train", 0)
    dataset_stats = dataset.calc_stats()

    fig, ax = plt.subplots()
    ax.bar(range(len(dataset_stats[2])), list(dataset_stats[2].values()))
    ax.set_xticks(range(len(dataset_stats[2])), labels=dataset_stats[2].keys(), rotation=90)
    ax.set_ylabel("class occurrences")
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
    dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

    dataset_train, dataset_test = DatasetTreeAI("ID", "train", 0).split(
        0.2,
        train_augmentations=make_base_train_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
        test_augmentations=make_base_test_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
        random_state=42,
    )

    n_samples = 5
    fig, ax = plt.subplots(4, n_samples, figsize=(n_samples * 4, 4 * 4))

    for i in range(n_samples):
        index_train = random.randint(0, len(dataset_train) - 1)
        index_test = random.randint(0, len(dataset_test) - 1)

        x_train, y_train = dataset_train[index_train]
        x_test, y_test = dataset_test[index_test]

        x_train = x_train.numpy().transpose(1, 2, 0)
        y_train = y_train.numpy()

        x_test = x_test.numpy().transpose(1, 2, 0)
        y_test = y_test.numpy()

        ax[0, i].set_title(f"train sample {index_train}")
        ax[0, i].imshow((x_train - x_train.min()) / (x_train.max() - x_train.min()))
        ax[0, i].axis("off")

        ax[2, i].set_title(f"test sample {index_test}")
        ax[2, i].imshow((x_test - x_test.min()) / (x_test.max() - x_test.min()))
        ax[2, i].axis("off")

        vmin = min(dataset_train.label_id_to_name.keys())
        vmax = max(dataset_train.label_id_to_name.keys())

        ax[1, i].imshow(y_train, vmin=vmin, vmax=vmax, cmap="viridis")
        ax[1, i].axis("off")

        ax[3, i].imshow(y_test, vmin=vmin, vmax=vmax, cmap="viridis")
        ax[3, i].axis("off")

    fig.tight_layout()
    fig.show()
