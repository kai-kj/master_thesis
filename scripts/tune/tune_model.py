# %%

from pathlib import Path
import time
import torch
import json

from scripts.train.loss import FocalTverskyLoss
from scripts.train.train import run_loop
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations, make_base_test_augmentations

# %%

model_path = Path("models/segformer_20260531_104250")
assert model_path.is_dir()

device = torch.device("cuda")
dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

for dist_name in ["china", "czechia", "iran", "mexico", "russia", "south_africa"]:
    sa = dist_name == "south_africa"
    mix_ratio = 1.0 if not sa else 0.0

    for k in [5, 10, 25, 50, 100]:
        print(f"==== TRAIN {dist_name} - {k} shot ====")

        dataset_train, dataset_valid = DatasetTreeAI(f"OOD_{dist_name}", "train", 0, dist_name == "south_africa") \
            .split(0.2, make_base_train_augmentations(), make_base_test_augmentations(), 42)

        dataset_train_extra, _ = DatasetTreeAI(f"ID", "train", 0) \
            .split(0.2, make_base_train_augmentations(), make_base_test_augmentations(), 42)

        if k > len(dataset_train):
            print("not enough data, skipping")
            continue

        # ---- unet ----

        from scripts.models.unet_res import UNetRes

        model = UNetRes(len(dataset_stats["y_counts"]), "resnet34").to(device)
        num_epochs = 25 if not sa else 50
        batch_size = 16
        lr_min = 1e-7 if not sa else 5e-7

        optimizer = torch.optim.AdamW([
            {"params": model.get_encoder_parameters(), "lr": 1e-6 if not sa else 5e-6},
            {"params": model.get_decoder_parameters(), "lr": 1e-5 if not sa else 5e-5},
            {"params": model.get_head_parameters(), "lr": 1e-5 if not sa else 5e-5},
        ])

        # ---- deeplab ----

        # from scripts.models.deeplab import DeepLabV3Plus
        #
        # model = DeepLabV3Plus(len(dataset_stats["y_counts"]), "resnet34").to(device)
        # num_epochs = 25 if not sa else 50
        # batch_size = 16
        # lr_min = 1e-7 if not sa else 5e-7
        #
        # optimizer = torch.optim.AdamW([
        #     {"params": model.get_encoder_parameters(), "lr": 1e-6 if not sa else 5e-6},
        #     {"params": model.get_decoder_parameters(), "lr": 1e-5 if not sa else 5e-5},
        #     {"params": model.get_head_parameters(), "lr": 1e-5 if not sa else 5e-5},
        # ])

        # ---- segformer ----

        # from scripts.models.segformer import SegFormer
        #
        # model = SegFormer(len(dataset_stats["y_counts"]), variant="b2").to(device)
        # num_epochs = 25 if not sa else 50
        # batch_size = 4
        # lr_min = 1e-7 if not sa else 5e-7
        #
        # optimizer = torch.optim.AdamW([
        #     {"params": model.get_encoder_parameters(), "lr": 1e-6 if not sa else 5e-6},
        #     {"params": model.get_decoder_parameters(), "lr": 1e-5 if not sa else 5e-5},
        # ])

        # ---- transunet ----

        # from scripts.models.transunet import TransUNet
        #
        # model = TransUNet(len(dataset_stats["y_counts"]), variant="b16").to(device)
        # num_epochs = 25 if not sa else 50
        # batch_size = 2
        # lr_min = 1e-7 if not sa else 5e-7
        #
        # optimizer = torch.optim.AdamW([
        #     {"params": model.get_encoder_parameters(), "lr": 1e-6 if not sa else 5e-6},
        #     {"params": model.get_decoder_parameters(), "lr": 1e-5 if not sa else 5e-5},
        #     {"params": model.get_head_parameters(), "lr": 1e-5 if not sa else 5e-5},
        # ])

        # ---- swin ----

        # from scripts.models.swin_fpn import SwinFPN
        #
        # model = SwinFPN(len(dataset_stats["y_counts"]), variant="b").to(device)
        # num_epochs = 25 if not sa else 50
        # batch_size = 4
        # lr_min = 1e-8 if not sa else 5e-8
        #
        # optimizer = torch.optim.AdamW([
        #     {"params": model.get_encoder_parameters(), "lr": 1e-7 if not sa else 5e-7},
        #     {"params": model.get_decoder_parameters(), "lr": 1e-6 if not sa else 5e-6},
        #     {"params": model.get_head_parameters(), "lr": 1e-6 if not sa else 5e-6},
        # ])

        # ---- run ----

        dataset_train.sample(k, random_state=42)
        dataset_train_extra.sample(int(len(dataset_train) * mix_ratio), random_state=42)

        if sa:
            print("changing n_classes")
            model.change_n_classes(len(dataset_stats["y_counts"]))

        run_loop(
            device, model, optimizer,
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min),
            FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.3).to(device),
            num_epochs,
            dataset_train.loader(batch_size, others=[dataset_train_extra]),
            dataset_valid.loader(batch_size),
            f"{model_path.as_posix()}_{dist_name}_{k}_shot_{time.strftime("%Y%m%d_%H%M%S")}",
            len(dataset_stats["y_counts"]),
            silent=True,
        )
