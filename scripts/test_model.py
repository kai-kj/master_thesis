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
import json

from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI
from scripts.util.seed import seed

# %%

model_path = Path("models/20260422_204630")
assert model_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

batch_size = 3
device = "cuda"

seed(42)

# %%

x_transform = tv_transforms.Compose([
    tv_transforms.Normalize(dataset_stats["x_mean"], dataset_stats["x_std"]),
])

dataset_test = DatasetTreeAI("ID", "test", 0, x_transform=x_transform)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# %%

model = UNet(len(dataset_stats["y_counts"]), norm="group").to(device)
model.load_state_dict(torch.load(model_path / "model.pt"))
model.eval()

# %%

n_samples = 5
fig, ax = plt.subplots(3, n_samples, figsize=(n_samples * 4, 3 * 4))

indices = [327, 57, 44, 287, 52]

for i in range(n_samples):
    # index = random.randint(0, len(dataset_test) - 1)
    index = indices[i]

    x, y_true = dataset_test[index]
    x_raw, _ = dataset_test.get_raw(index)

    y_pred = model(x.to(device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    y_pred = y_pred.transpose(1, 2, 0).argmax(2)[:, :, np.newaxis]

    x = x.numpy().transpose(1, 2, 0)
    x_raw = x_raw.numpy().transpose(1, 2, 0)
    y_true = y_true.numpy()

    ax[0, i].imshow((x - x.min()) / (x.max() - x.min()))
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
fig.savefig(model_path / "results.png")
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

confusion_matrix = skm.confusion_matrix(
    ys_true, ys_pred,
    labels=list(dataset_test.label_name_to_id.values()), normalize="true"
)

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
fig.savefig(model_path / "confusion_matrix.png")
fig.show()

# %%

metrics = []

for metric in ["micro", "macro", "weighted"]:
    print(f"computing {metric} metrics")

    p = skm.precision_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore
    r = skm.recall_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore
    f1 = skm.f1_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore
    iou = skm.jaccard_score(ys_true.flatten(), ys_pred.flatten(), average=metric, zero_division=0.0)  # type: ignore

    metrics.append([metric, p, r, f1, iou])

metrics = pd.DataFrame(metrics, columns=["metric", "precision", "recall", "f1", "iou"])

print(metrics)

metrics.to_csv(model_path / "metrics.csv")
