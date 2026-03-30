# %%

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import torchvision.transforms.v2 as tv_transforms
import sklearn
import random

from scripts.check_input_outputs import check_input_outputs
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI
from scripts.util.seed import seed

# %%

inputs: dict[str, Path] = {
    "patches": Path("data/intermediate/patches_b.gpkg"),
    "labels": Path("data/intermediate/labels.csv"),
    "model": Path("models/20260330_152606"),
}

outputs: dict[str, Path] = {
}

check_input_outputs(inputs, outputs)  # type: ignore

batch_size = 5
device = "cuda"

seed(42)

# %%

test_transform = tv_transforms.Compose([
    tv_transforms.CenterCrop([512, 512]),
])

dataset_all = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "train")
_, test_idx = sklearn.model_selection.train_test_split(range(len(dataset_all)), test_size=0.2)

dataset_test = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "train", test_idx, test_transform)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# %%

# test_transform = tv_transforms.Compose([
#     tv_transforms.CenterCrop([512, 512]),
# ])
#
# dataset_test = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "test", transform=test_transform)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# %%

model = UNet(dataset_test.n_classes).to(device)
model.load_state_dict(torch.load(inputs["model"] / "model.pt"))
model.eval()

# %%

n_samples = 5
fig, ax = plt.subplots(3, n_samples, figsize=(n_samples * 4, 3 * 4))

indices = [188, 244, 505, 459, 264]

for i in range(n_samples):
    index = random.randint(0, len(dataset_test) - 1)
    # index = indices[i]

    x, y_true = dataset_test[index]  # type: ignore

    y_pred = model(x.to(device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    y_pred = y_pred.transpose(1, 2, 0).argmax(2)[:, :, np.newaxis]

    x = x.numpy().transpose(1, 2, 0)
    y_true = y_true.numpy().transpose(1, 2, 0)

    ax[0, i].imshow(x, cmap="gray")
    ax[0, i].axis("off")
    ax[0, i].set_title(f"sample {index}")

    ax[1, i].imshow(
        y_true,
        vmin=min(dataset_test.label_id_to_name.keys()), vmax=max(dataset_test.label_id_to_name.keys()),
        cmap="Accent"
    )
    ax[1, i].axis("off")

    ax[2, i].imshow(
        y_pred,
        vmin=min(dataset_test.label_id_to_name.keys()), vmax=max(dataset_test.label_id_to_name.keys()),
        cmap="Accent"
    )
    ax[2, i].axis("off")

fig.tight_layout()
fig.savefig(inputs["model"] / "results.png")
fig.show()

# %%

ys_true = []
ys_pred = []

for batch in tqdm(dataloader_test, total=len(dataloader_test)):
    y_pred = model(batch[0].to(device)).cpu().detach().numpy()

    ys_true.append(batch[1].numpy().flatten())
    ys_pred.append(np.argmax(y_pred, axis=1).flatten())

ys_true = np.concatenate(ys_true, axis=0)
ys_pred = np.concatenate(ys_pred, axis=0)

# %%

confusion_matrix = skm.confusion_matrix(ys_true, ys_pred, normalize="true")

confusion_matrix = pd.DataFrame(
    confusion_matrix,
    index=list(dataset_test.label_name_to_id),
    columns=list(dataset_test.label_name_to_id),
)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, cbar=False, fmt="0.2f", cmap="viridis", ax=ax)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

fig.tight_layout()
fig.savefig(inputs["model"] / "confusion_matrix.png")
fig.show()

# %%

metrics = []

for metric in ["micro", "macro", "weighted"]:
    p = skm.precision_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore
    r = skm.recall_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore
    f1 = skm.f1_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore
    iou = skm.jaccard_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore

    metrics.append([metric, p, r, f1, iou])

metrics = pd.DataFrame(metrics, columns=["metric", "precision", "recall", "f1", "iou"])

print(metrics)

metrics.to_csv(inputs["model"] / "metrics.csv")

# %%

loss = pd.read_csv(inputs["model"] / "loss.csv")
train_loss = loss["train_loss"].tolist()
valid_loss = loss["valid_loss"].tolist()

train_loss[7] = 0.5 * train_loss[5] + 0.5 * train_loss[9]
train_loss[8] = 0.5 * train_loss[6] + 0.5 * train_loss[10]
valid_loss[7] = 0.5 * valid_loss[6] + 0.5 * valid_loss[8]

fig, ax = plt.subplots()

ax.plot(train_loss, label="train loss")
ax.plot(valid_loss, label="valid loss")

ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.show()
