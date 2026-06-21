# %%

from pathlib import Path
import time
import torch
import json

from scripts.train.loss import FocalTverskyLoss, calc_class_weights
from scripts.train.train import run_loop
from scripts.models.unet_scratch import UNet
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations_strong, make_base_test_augmentations

# %%

model_path = Path("models/unet_ftv_20260505_023132")
assert model_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

for dist_name in ["china", "czechia", "iran", "mexico", "russia", "south_africa"]:
    mix_ratio = 1.0 if dist_name != "south_africa" else 0.0

    print(f"==== TRAIN {dist_name} ====")

    dataset_train, dataset_valid = DatasetTreeAI(f"OOD_{dist_name}", "train", 0, dist_name == "south_africa").split(
        0.2,
        train_augmentations=make_base_train_augmentations_strong(dataset_stats["x_mean"], dataset_stats["x_std"]),
        test_augmentations=make_base_test_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
        random_state=42,
    )

    dataset_train_extra, _ = DatasetTreeAI(f"ID", "train", 0).split(
        0.2,
        train_augmentations=make_base_train_augmentations_strong(dataset_stats["x_mean"], dataset_stats["x_std"]),
        test_augmentations=make_base_test_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
        random_state=42,
    )

    dataset_train_extra.sample(int(len(dataset_train) * mix_ratio), random_state=42)

    batch_size = 6

    dataloader_train = dataset_train.loader(batch_size, others=[dataset_train_extra])
    dataloader_valid = dataset_valid.loader(batch_size)

    device = torch.device("cuda")

    base_lr = 5e-5 if dist_name != "south_africa" else 1e-4
    num_epochs = 20

    model = UNet.load(model_path / "model.json", model_path / "model.pt").to(device)

    if dist_name == "south_africa":
        model.change_n_classes(len(dataset_stats["y_counts"]))

    # for module in model.get_encoder_modules():
    #     for param in module.parameters():
    #         param.requires_grad = False

    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr * 0.01)

    run_loop(
        device, model, optimizer, scheduler, loss_fn,
        num_epochs, dataloader_train, dataloader_valid,
        f"{model_path.as_posix()}_{dist_name}_{time.strftime("%Y%m%d_%H%M%S")}",
        len(dataset_stats["y_counts"])
    )
