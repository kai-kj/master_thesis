# %%

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import sklearn.metrics as skm
# import seaborn as sns
import pandas as pd

from scripts.check_input_outputs import check_input_outputs
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI

# %%

inputs: dict[str, Path] = {
    "patches": Path("data/intermediate/patches_b.gpkg"),
    "labels": Path("data/intermediate/labels.csv"),
    "model": Path("models/20260323_110305"),
}

outputs: dict[str, Path] = {
}

check_input_outputs(inputs, outputs)

device = "cuda"

# %%

generator = torch.Generator().manual_seed(42)
dataset_all = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "train")
dataset_train, dataset_valid = torch.utils.data.random_split(dataset_all, [0.8, 0.2], generator)

# %%

model = UNet(dataset_all.n_classes).to(device)
model.load_state_dict(torch.load(inputs["model"] / "model.pt"))
model.eval()

# %%

n_samples = 6
fig, ax = plt.subplots(3, n_samples, figsize=(n_samples * 4, 3 * 4))

indices = [286, 164, 188, 131, 244, 277]

for i in range(n_samples):
    # index = random.randint(0, len(dataset_valid) - 1)
    index = indices[i]

    x, y_true = dataset_valid[index]  # type: ignore

    y_pred = model(x.to(device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    y_pred = y_pred.transpose(1, 2, 0).argmax(2)[:, :, np.newaxis]

    x = x.numpy().transpose(1, 2, 0)
    y_true = y_true.numpy().transpose(1, 2, 0)

    ax[0, i].imshow(x, cmap="gray")
    ax[0, i].axis("off")
    ax[0, i].set_title(f"sample {index}")

    vmin = min(dataset_all.label_id_to_name.keys())

    ax[1, i].imshow(
        y_true,
        vmin=min(dataset_all.label_id_to_name.keys()), vmax=max(dataset_all.label_id_to_name.keys()),
        cmap="viridis"
    )
    ax[1, i].axis("off")

    ax[2, i].imshow(
        y_pred,
        vmin=min(dataset_all.label_id_to_name.keys()), vmax=max(dataset_all.label_id_to_name.keys()),
        cmap="viridis"
    )
    ax[2, i].axis("off")

fig.tight_layout()
fig.show()

# %%

# ys_true = []
# ys_pred = []
#
# for batch in test_dataloader:
#     y_pred = model(batch[0].to(device)).cpu().detach().numpy()
#
#     ys_true.append([test_dataset.index2class[i] for i in np.argmax(batch[1].numpy(), axis=1)])
#     ys_pred.append([test_dataset.index2class[i] for i in np.argmax(y_pred, axis=1)])
#
# ys_true = [y for ys in ys_true for y in ys]
# ys_pred = [y for ys in ys_pred for y in ys]
#
# # %%
#
# confusion_matrix = skm.confusion_matrix(ys_true, ys_pred)
#
# confusion_matrix = pd.DataFrame(
#     confusion_matrix,
#     index=list(test_dataset.class2index),
#     columns=list(test_dataset.class2index),
# )
#
# fig, ax = plt.subplots(figsize=(18, 18))
# sns.heatmap(confusion_matrix, annot=True, cbar=False, fmt="d", ax=ax)
# fig.tight_layout()
# fig.savefig(model_path / "confusion_matrix.png")
# fig.show()
