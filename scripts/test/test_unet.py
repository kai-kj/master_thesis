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

from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI, make_base_test_augmentations
from scripts.util.seed import seed

# %%

model_path = Path("models/unet_ftv_20260505_023132_south_africa_20260505_135708")
assert model_path.is_dir()

dist_name = "ID"
# dist_name = "china"
# dist_name = "czechia"
# dist_name = "iran"
# dist_name = "mexico"
# dist_name = "russia"
# dist_name = "south_africa"

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

batch_size = 3
device = "cuda"

seed(42)

# %%

dataset_test = DatasetTreeAI(
    "ID" if dist_name == "ID" else f"OOD_{dist_name}", "test", 0, dist_name == "south_africa",
    augmentations=make_base_test_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# %%

model = UNet.load(model_path / "model.json", model_path / "model.pt").to(device)
model.eval()

# %%

n_samples = 5
fig, ax = plt.subplots(3, n_samples, figsize=(n_samples * 4, 3 * 4))

indices = [47, 71, 111, 216, 346] if dist_name == "ID" else np.arange(0, len(dataset_test), len(dataset_test) // 5)

for i in range(n_samples):
    index = indices[i]
    # index = random.randint(0, len(dataset_test))

    x, y_true = dataset_test[index]

    y_pred = model(x.to(device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
    y_pred = y_pred.transpose(1, 2, 0).argmax(2)[:, :, np.newaxis]

    x = x.numpy().transpose(1, 2, 0)
    y_true = y_true.numpy()

    ax[0, i].imshow((x - x.min()) / (x.max() - x.min()))
    ax[0, i].axis("off")
    ax[0, i].set_title(f"sample {index}")

    ax[1, i].imshow(
        y_true,
        vmin=-1, vmax=max(dataset_test.label_id_to_name.keys()),
        cmap="Accent"
    )
    ax[1, i].axis("off")

    ax[2, i].imshow(
        y_pred,
        vmin=-1, vmax=max(dataset_test.label_id_to_name.keys()),
        cmap="Accent"
    )
    ax[2, i].axis("off")

fig.tight_layout()
fig.savefig(model_path / f"results_{dist_name.lower()}.png")
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
fig.savefig(model_path / f"confusion_matrix_{dist_name.lower()}.png")
fig.show()

# %%

metrics = []

for metric in ["micro", "macro", "weighted"]:
    print(f"computing {metric} metrics")
    p, r, f1, _ = skm.precision_recall_fscore_support(
        ys_true.flatten(), ys_pred.flatten(),
        average=metric, zero_division=0.0, labels=np.unique(ys_true)  # type: ignore
    )
    iou = skm.jaccard_score(
        ys_true.flatten(),
        ys_pred.flatten(),
        average=metric, zero_division=0.0, labels=np.unique(ys_true)  # type: ignore
    )
    metrics.append([metric, p, r, f1, iou])

metrics = pd.DataFrame(metrics, columns=["metric", "precision", "recall", "f1", "iou"])

print(metrics)

metrics.to_csv(model_path / f"metrics_{dist_name.lower()}.csv")
