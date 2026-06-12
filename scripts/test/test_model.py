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

from scripts.models.model import Model
from scripts.dataset import DatasetTreeAI, make_base_test_augmentations
from scripts.util.seed import seed


# %%

def test_model(
        model_path: Path, dist_name: str, batch_size: int, device: str,
        x_mean: tuple[float, float, float], x_std: tuple[float, float, float],
):
    assert model_path.exists()

    seed(42)

    dataset_test = DatasetTreeAI(
        "ID" if dist_name == "ID" else f"OOD_{dist_name}", "test", 0, dist_name == "south_africa",
        augmentations=make_base_test_augmentations(x_mean, x_std),
    )

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    model = Model.load(model_path / "model.json", model_path / "model.pt").to(device)
    model.eval()

    print(f"loaded model {model._name}")

    n_samples = 4
    fig, ax = plt.subplots(3, n_samples, figsize=(n_samples * 4, 3 * 4))

    # indices = [47, 71, 111, 216, 346] if dist_name == "ID" else np.arange(0, len(dataset_test), len(dataset_test) // n_samples)
    indices = [47, 111, 216, 346] \
        if dist_name == "ID" \
        else np.arange(0, len(dataset_test), len(dataset_test) // n_samples)

    for i in range(n_samples):
        index = indices[i]
        # index = random.randint(0, len(dataset_test))

        x, y_true = dataset_test[index]
        x_raw = dataset_test.get_xy(index, apply_augmentations=False)[0]

        y_pred = model(x.to(device).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        y_pred = y_pred.transpose(1, 2, 0).argmax(2)[:, :, np.newaxis]

        x_raw = x_raw.numpy().transpose(1, 2, 0)
        y_true = y_true.numpy()

        ax[0, i].imshow((x_raw - x_raw.min()) / (x_raw.max() - x_raw.min()))
        ax[0, i].axis("off")
        ax[0, i].set_title(f"sample {index}", fontsize=20)

        ax[1, i].imshow(y_true, vmin=-1, vmax=max(dataset_test.label_id_to_name.keys()), cmap="Accent")
        ax[1, i].axis("off")

        ax[2, i].imshow(y_pred, vmin=-1, vmax=max(dataset_test.label_id_to_name.keys()), cmap="Accent")
        ax[2, i].axis("off")

    ax[0, 0].text(
        -0.05, 0.5, "x", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=20
    )
    ax[1, 0].text(
        -0.05, 0.5, "y_true", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="center", fontsize=20
    )
    ax[2, 0].text(
        -0.05, 0.5, "y_pred", transform=ax[2, 0].transAxes, rotation=90, va="center", ha="center", fontsize=20
    )

    fig.tight_layout()
    fig.savefig(model_path / f"results_{dist_name.lower()}.png")
    fig.show()

    ys_true = []
    ys_pred = []

    for batch in tqdm(dataloader_test, total=len(dataloader_test)):
        y_pred = model(batch[0].to(device)).cpu().detach().numpy()
        ys_true.append(batch[1].numpy().flatten())
        ys_pred.append(np.argmax(y_pred, axis=1).flatten())

    ys_true = np.concatenate(ys_true, axis=0)
    ys_pred = np.concatenate(ys_pred, axis=0)

    confusion_matrix = skm.confusion_matrix(
        ys_true, ys_pred,
        labels=list(dataset_test.label_name_to_id.values()), normalize="true"
    )

    confusion_matrix = pd.DataFrame(
        confusion_matrix,
        index=list(dataset_test.label_name_to_id),
        columns=list(dataset_test.label_name_to_id),
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.heatmap(confusion_matrix, annot=True, cbar=False, fmt="0.2f", cmap="viridis", ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0 + 30)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90 - 30)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    fig.tight_layout()
    fig.savefig(model_path / f"confusion_matrix_{dist_name.lower()}.png")
    fig.show()

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

    metrics.to_csv(model_path / f"metrics_{dist_name.lower()}.csv", index=False)


def test_models(
        model_dists: list[tuple[Path, str]], batch_size: int, device: str,
        x_mean: tuple[float, float, float], x_std: tuple[float, float, float],
):
    for model_path, dist_name in model_dists:
        test_model(model_path, dist_name, batch_size, device, x_mean, x_std)


# %%

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# test_model(
#     Path("models/segformer_20260530_164843"),
#     "ID",
#     2, "cuda",
#     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
# )

# test_models(
#     [
#         (Path("models/transunet_20260527_181604"), "ID"),
#         (Path("models/transunet_20260527_181604"), "china"),
#         (Path("models/transunet_20260527_181604"), "czechia"),
#         (Path("models/transunet_20260527_181604"), "iran"),
#         (Path("models/transunet_20260527_181604"), "mexico"),
#         (Path("models/transunet_20260527_181604"), "russia"),
#         (Path("models/transunet_20260527_181604_china_20260529_125823"), "ID"),
#         (Path("models/transunet_20260527_181604_china_20260529_125823"), "china"),
#         (Path("models/transunet_20260527_181604_czechia_20260529_130013"), "ID"),
#         (Path("models/transunet_20260527_181604_czechia_20260529_130013"), "czechia"),
#         (Path("models/transunet_20260527_181604_iran_20260529_131858"), "ID"),
#         (Path("models/transunet_20260527_181604_iran_20260529_131858"), "iran"),
#         (Path("models/transunet_20260527_181604_mexico_20260529_132655"), "ID"),
#         (Path("models/transunet_20260527_181604_mexico_20260529_132655"), "mexico"),
#         (Path("models/transunet_20260527_181604_russia_20260529_133130"), "ID"),
#         (Path("models/transunet_20260527_181604_russia_20260529_133130"), "russia"),
#         (Path("models/transunet_20260527_181604_south_africa_20260529_134122"), "ID"),
#         (Path("models/transunet_20260527_181604_south_africa_20260529_134122"), "south_africa"),
#     ],
#     2, "cuda",
#     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
# )

# test_models(
#     [
#         (Path("models/swin_fpn_20260529_145014"), "ID"),
#         (Path("models/swin_fpn_20260529_145014"), "china"),
#         (Path("models/swin_fpn_20260529_145014"), "czechia"),
#         (Path("models/swin_fpn_20260529_145014"), "iran"),
#         (Path("models/swin_fpn_20260529_145014"), "mexico"),
#         (Path("models/swin_fpn_20260529_145014"), "russia"),
#         (Path("models/swin_fpn_20260529_145014_china_20260529_173831"), "ID"),
#         (Path("models/swin_fpn_20260529_145014_china_20260529_173831"), "china"),
#         (Path("models/swin_fpn_20260529_145014_czechia_20260529_173955"), "ID"),
#         (Path("models/swin_fpn_20260529_145014_czechia_20260529_173955"), "czechia"),
#         (Path("models/swin_fpn_20260529_145014_iran_20260529_175350"), "ID"),
#         (Path("models/swin_fpn_20260529_145014_iran_20260529_175350"), "iran"),
#         (Path("models/swin_fpn_20260529_145014_mexico_20260529_175952"), "ID"),
#         (Path("models/swin_fpn_20260529_145014_mexico_20260529_175952"), "mexico"),
#         (Path("models/swin_fpn_20260529_145014_russia_20260529_180330"), "ID"),
#         (Path("models/swin_fpn_20260529_145014_russia_20260529_180330"), "russia"),
#         (Path("models/swin_fpn_20260529_145014_south_africa_20260529_181057"), "south_africa"),
#     ],
#     2, "cuda",
#     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
# )

test_models(
    [
        (Path("models/segformer_20260531_104250"), "ID"),
        (Path("models/segformer_20260531_104250"), "china"),
        (Path("models/segformer_20260531_104250"), "czechia"),
        (Path("models/segformer_20260531_104250"), "iran"),
        (Path("models/segformer_20260531_104250"), "mexico"),
        (Path("models/segformer_20260531_104250"), "russia"),
        (Path("models/segformer_20260531_104250_china_20260531_164020"), "ID"),
        (Path("models/segformer_20260531_104250_china_20260531_164020"), "china"),
        (Path("models/segformer_20260531_104250_czechia_20260531_164110"), "ID"),
        (Path("models/segformer_20260531_104250_czechia_20260531_164110"), "czechia"),
        (Path("models/segformer_20260531_104250_iran_20260531_164717"), "ID"),
        (Path("models/segformer_20260531_104250_iran_20260531_164717"), "iran"),
        (Path("models/segformer_20260531_104250_mexico_20260531_165002"), "ID"),
        (Path("models/segformer_20260531_104250_mexico_20260531_165002"), "mexico"),
        (Path("models/segformer_20260531_104250_russia_20260531_165142"), "ID"),
        (Path("models/segformer_20260531_104250_russia_20260531_165142"), "russia"),
        (Path("models/segformer_20260531_104250_south_africa_20260531_165502"), "south_africa"),
    ],
    2, "cuda",
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
)
