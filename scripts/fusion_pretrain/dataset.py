# %%

import random
from pathlib import Path
import json
from typing import Literal

import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import geopandas as gpd
import cv2


# %%

class DatasetClimateSoil(torch.utils.data.Dataset):
    TREEAI_SEGMENTATION_PATH = Path("/home/kai/Thesis/data/treeai_segmentation/")

    def __init__(
            self,
            train_or_test: str, split_n: int,
            include_clim: bool = False, include_soil: bool = False,
            indices: None | list[int] = None,
    ):
        super(DatasetClimateSoil, self).__init__()

        assert self.TREEAI_SEGMENTATION_PATH.exists()

        patches_y_path = self.TREEAI_SEGMENTATION_PATH / "patches_pretrain_y.gpkg"
        assert patches_y_path.is_file()

        patches_x_clim_path = self.TREEAI_SEGMENTATION_PATH / "patches_pretrain_x_clim.gpkg"
        assert patches_x_clim_path.is_file()

        patches_x_soil_path = self.TREEAI_SEGMENTATION_PATH / "patches_pretrain_x_soil.gpkg"
        assert patches_x_soil_path.is_file()

        # for when calling .split()
        self._config = {"train_or_test": train_or_test, "split_n": split_n}

        self._data = gpd.read_file(patches_y_path)
        self._data = self._data[self._data[f"split_{split_n}"] == train_or_test]

        if include_clim:
            self._data = pd.merge(self._data, gpd.read_file(patches_x_clim_path), on="patch_name")

        if include_soil:
            self._data = pd.merge(self._data, gpd.read_file(patches_x_soil_path), on="patch_name")

        if indices is not None:
            self._data = self._data.iloc[indices]

        self._data = self._data.drop(columns=[c for c in self._data.columns if "geometry" in c])

        self.features = [f for f in self._data.columns if f.startswith("x_")]
        self.labels = [l for l in self._data.columns if l.startswith("y_")]

    def split(
            self,
            test_size: float,
            random_state: int | None = None
    ) -> tuple["DatasetClimateSoil", "DatasetClimateSoil"]:
        train_idx, test_idx = sklearn.model_selection.train_test_split(
            range(len(self)), test_size=test_size, random_state=random_state
        )

        train_dataset = DatasetClimateSoil(indices=train_idx, **self._config)
        test_dataset = DatasetClimateSoil(indices=test_idx, **self._config)

        return train_dataset, test_dataset

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._data.iloc[idx]

        x = torch.Tensor(row[self.features].to_numpy(dtype=np.float64))
        y = torch.Tensor(row[self.labels].to_numpy(dtype=np.float64))

        return x, y


# %%


if __name__ == "__main__":
    dataset = DatasetClimateSoil("train", 0, include_clim=True)

    print(f"x shape: {dataset[0][0].shape}")
    print(f"y shape: {dataset[0][1].shape}")
