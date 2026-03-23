# %%

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import sklearn.metrics as skm
import seaborn as sns
from tqdm import tqdm
import pandas as pd

from scripts.check_input_outputs import check_input_outputs
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI

# %%

inputs: dict[str, Path] = {
    "patches": Path("data/intermediate/patches_b.gpkg"),
    "labels": Path("data/intermediate/labels.csv"),
    "model": Path("models/20260323_154004"),
}

outputs: dict[str, Path] = {
}

check_input_outputs(inputs, outputs)

batch_size = 5
device = "cuda"

# %%

generator = torch.Generator().manual_seed(42)
dataset_all = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "train")
dataset_train, dataset_valid = torch.utils.data.random_split(dataset_all, [0.8, 0.2], generator)

dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

# %%

model = UNet(dataset_all.n_classes).to(device)
model.load_state_dict(torch.load(inputs["model"] / "model.pt"))
model.eval()

# %%

n_samples = 5
fig, ax = plt.subplots(3, n_samples, figsize=(n_samples * 4, 3 * 4))

indices = [188, 244, 505, 459, 264]

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

    ax[1, i].imshow(
        y_true,
        vmin=min(dataset_all.label_id_to_name.keys()), vmax=max(dataset_all.label_id_to_name.keys()),
        cmap="Accent"
    )
    ax[1, i].axis("off")

    ax[2, i].imshow(
        y_pred,
        vmin=min(dataset_all.label_id_to_name.keys()), vmax=max(dataset_all.label_id_to_name.keys()),
        cmap="Accent"
    )
    ax[2, i].axis("off")

fig.tight_layout()
fig.show()

# %%

ys_true = []
ys_pred = []

for batch in tqdm(dataloader_valid, total=len(dataloader_valid)):
    y_pred = model(batch[0].to(device)).cpu().detach().numpy()

    ys_true.append(batch[1].numpy().flatten())
    ys_pred.append(np.argmax(y_pred, axis=1).flatten())

ys_true = np.concatenate(ys_true, axis=0)
ys_pred = np.concatenate(ys_pred, axis=0)

# %%

confusion_matrix = skm.confusion_matrix(ys_true, ys_pred, normalize="true")

confusion_matrix = pd.DataFrame(
    confusion_matrix,
    index=list(dataset_all.label_name_to_id),
    columns=list(dataset_all.label_name_to_id),
)

# %%

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, cbar=False, fmt="0.2f", cmap="viridis", ax=ax)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

fig.tight_layout()
fig.savefig(inputs["model"] / "confusion_matrix.png")
fig.show()
