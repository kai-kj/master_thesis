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
from matplotlib.axes import Axes
import random
from scripts.models.model import Model
from scripts.dataset import DatasetTreeAI, make_base_test_augmentations

device = torch.device("cpu")

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

def make_preview(
        dataset: DatasetTreeAI, models: dict[str, Model],
        sample_idxs: list[int], add_sample_names: bool = False,
        path: Path | None = None,
):
    img_size = 2.5

    fig, ax = plt.subplots(
        len(sample_idxs), len(models) + 3,
        figsize=((len(models) + 2.25) * img_size, len(sample_idxs) * img_size),
        gridspec_kw={
            "width_ratios": [1, 1, 0.25] + [1] * len(models)
        },
    )

    for i, sample_idx in enumerate(sample_idxs):
        x, _, y_true = dataset[sample_idx]
        x_raw = dataset.get_xy(sample_idx, apply_augmentations=False)[0]

        x_raw_np = x_raw.numpy().transpose(1, 2, 0)
        y_true_np = y_true.numpy()

        ax[i, 0].imshow((x_raw_np - x_raw_np.min()) / (x_raw_np.max() - x_raw_np.min()))
        ax[i, 0].axis("off")

        ax[i, 1].imshow(y_true_np, vmin=-1, vmax=max(dataset.label_id_to_name.keys()), cmap="Accent")
        ax[i, 1].axis("off")

        ax[i, 2].axis("off")

        if i == 0:
            ax[i, 0].set_title("input", fontsize=18)
            ax[i, 1].set_title("ground truth", fontsize=18)

        for j, (model_name, model) in enumerate(models.items()):
            model.eval()
            y_pred = model(x.to(device).unsqueeze(0), None).squeeze(0)
            y_pred_np = y_pred.cpu().detach().numpy().transpose(1, 2, 0).argmax(2)[:, :, np.newaxis]

            ax[i, j + 3].imshow(y_pred_np, vmin=-1, vmax=max(dataset.label_id_to_name.keys()), cmap="Accent")
            ax[i, j + 3].axis("off")

            if i == 0: ax[i, j + 3].set_title(f"{model_name}", fontsize=18)

        if add_sample_names:
            ax[i, 0].text(
                -0.1, 0.5, f"sample {sample_idxs[i]}",
                transform=ax[i, 0].transAxes, rotation=90, va="center", ha="center", fontsize=18
            )

    fig.tight_layout()

    if path is not None:
        fig.savefig(path)

    fig.show()


# %%

core_model_paths = {
    "UNet": Path("models/unet_split_0_20260612_122705"),
    "DeepLabV3+": Path("models/deeplab_split_0_20260612_135725"),
    "Segformer": Path("models/segformer_split_0_20260612_145904"),
    "TransUNet": Path("models/transunet_split_0_20260612_170208"),
    "SwinFPN": Path("models/swin_split_0_20260612_203922"),
}

core_models = {
    model_name: Model.load(model_path / "model.json", model_path / "model.pt").to(device)
    for model_name, model_path in core_model_paths.items()
}

dataset = DatasetTreeAI("ID", "test", 0, augmentations=make_base_test_augmentations())

make_preview(
    dataset, core_models, [47, 111, 346],
    False, Path("figures/preview_core.png")
)

# %%

# dist_names = ["austria", "china", "czechia", "iran", "mexico", "russia", "south_africa"]
random.seed(1)

# [random.randint(0, 10) for _ in range(5)]
# list(range(8))
# 12, 13, 18

dist_names = {
    "austria": [25, 19, 27],
    "china": [4, 0, 2],
    "czechia": [26, 24, 48],
    "iran": [24, 1, 5],
    "mexico": [0, 2, 9],
    "russia": [25, 13, 23],
    "south_africa": [12, 13, 18],
}

for dist_name, indices in dist_names.items():
    print(f"processing {dist_name}")

    dataset_test = DatasetTreeAI(
        f"OOD_{dist_name}", "test", 0, dist_name == "south_africa",
        augmentations=make_base_test_augmentations(),
    )

    ft_models = {}

    for model_name, core_model_path in core_model_paths.items():
        ft_model_path = None

        for k in [1, 5, 25, 125]:
            candidates = list(core_model_path.parent.glob(f"{core_model_path.name}_{dist_name}_{k}_shot_*"))
            if len(candidates) > 0: ft_model_path = candidates[0]

        assert ft_model_path is not None

        print(f"  loading {ft_model_path}")
        ft_models[model_name] = Model.load(ft_model_path / "model.json", ft_model_path / "model.pt").to(device)

    make_preview(
        dataset_test, ft_models, indices,
        False, Path(f"figures/preview_{dist_name}.png")
    )

# %%

fig, ax = plt.subplots(
    len(dist_names), len(core_model_paths) + 3,
    figsize=((len(core_model_paths) + 2.25) * 2.5, len(dist_names) * 2.5),
    gridspec_kw={
        "width_ratios": [1, 1, 0.25] + [1] * len(core_model_paths)
    },
)

for i, (dist_name, sample_idxs) in enumerate(dist_names.items()):
    print(f"processing {dist_name}")

    dataset = DatasetTreeAI(
        f"OOD_{dist_name}", "test", 0, dist_name == "south_africa",
        augmentations=make_base_test_augmentations(),
    )

    x, _, y_true = dataset[sample_idxs[0]]
    x_raw = dataset.get_xy(sample_idxs[0], apply_augmentations=False)[0]

    x_raw_np = x_raw.numpy().transpose(1, 2, 0)
    y_true_np = y_true.numpy()

    ax[i, 0].imshow((x_raw_np - x_raw_np.min()) / (x_raw_np.max() - x_raw_np.min()))
    ax[i, 0].axis("off")

    ax[i, 1].imshow(y_true_np, vmin=-1, vmax=max(dataset.label_id_to_name.keys()), cmap="Accent")
    ax[i, 1].axis("off")

    ax[i, 2].axis("off")

    if i == 0:
        ax[i, 0].set_title("input", fontsize=18)
        ax[i, 1].set_title("ground truth", fontsize=18)

    for j, (model_name, core_model_path) in enumerate(core_model_paths.items()):
        ft_model_path = None

        for k in [1, 5, 25, 125]:
            candidates = list(core_model_path.parent.glob(f"{core_model_path.name}_{dist_name}_{k}_shot_*"))
            if len(candidates) > 0: ft_model_path = candidates[0]

        assert ft_model_path is not None

        print(f"  loading {ft_model_path}")
        model = Model.load(ft_model_path / "model.json", ft_model_path / "model.pt").to(device)

        model.eval()
        y_pred = model(x.to(device).unsqueeze(0), None).squeeze(0)
        y_pred_np = y_pred.cpu().detach().numpy().transpose(1, 2, 0).argmax(2)[:, :, np.newaxis]

        ax[i, j + 3].imshow(y_pred_np, vmin=-1, vmax=max(dataset.label_id_to_name.keys()), cmap="Accent")
        ax[i, j + 3].axis("off")

        if i == 0: ax[i, j + 3].set_title(f"{model_name}", fontsize=18)

    ax[i, 0].text(
        -0.1, 0.5, dist_name.replace("_", " "),
        transform=ax[i, 0].transAxes, rotation=90, va="center", ha="center", fontsize=18
    )

fig.tight_layout()

fig.savefig(Path("figures/preview_all.png"))
fig.show()
