# %%

from pathlib import Path
import time
import torch
import json

from scripts.train.loss import FocalTverskyLoss
from scripts.train.train import run_loop
from scripts.models.transunet import TransUNet
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations_strong, make_base_test_augmentations

# %%

model_path = Path("models/transunet_20260527_181604")
assert model_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

for dist_name in ["china", "czechia", "iran", "mexico", "russia", "south_africa"]:
    mix_ratio = 1.0 if dist_name != "south_africa" else 0.0
    # mix_ratio = 0.0

    print(f"==== TRAIN {dist_name} ====")

    dataset_train, dataset_valid = DatasetTreeAI(f"OOD_{dist_name}", "train", 0, dist_name == "south_africa").split(
        0.2,
        train_augmentations=make_base_train_augmentations_strong((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        test_augmentations=make_base_test_augmentations((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        random_state=42,
    )

    dataset_train_extra, _ = DatasetTreeAI(f"ID", "train", 0).split(
        0.2,
        train_augmentations=make_base_train_augmentations_strong((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        test_augmentations=make_base_test_augmentations((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        random_state=42,
    )

    dataset_train_extra.sample(int(len(dataset_train) * mix_ratio), random_state=42)

    batch_size = 2

    dataloader_train = dataset_train.loader(batch_size, others=[dataset_train_extra])
    dataloader_valid = dataset_valid.loader(batch_size)

    device = torch.device("cuda")

    base_lr_encoder = 1e-6 if dist_name != "south_africa" else 5e-6
    base_lr_decoder = 1e-5 if dist_name != "south_africa" else 5e-5
    base_lr_head = 1e-5 if dist_name != "south_africa" else 5e-5
    num_epochs = 25 if dist_name != "south_africa" else 50

    model = TransUNet.load(model_path / "model.json", model_path / "model.pt").to(device)

    if dist_name == "south_africa":
        print("changing n_classes")
        model.change_n_classes(len(dataset_stats["y_counts"]))

    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.3).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.get_encoder_parameters(), "lr": base_lr_encoder},
        {"params": model.get_decoder_parameters(), "lr": base_lr_decoder},
        {"params": model.get_head_parameters(), "lr": base_lr_head},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min(base_lr_encoder, base_lr_decoder, base_lr_head) * 0.1
    )

    run_loop(
        device, model, optimizer, scheduler, loss_fn,
        num_epochs, dataloader_train, dataloader_valid,
        f"{model_path.as_posix()}_{dist_name}_{time.strftime("%Y%m%d_%H%M%S")}",
        len(dataset_stats["y_counts"])
    )
