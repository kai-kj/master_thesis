# %%

from pathlib import Path
import time
import torch
import json

from scripts.train.loss import FocalTverskyLoss
from scripts.train.train import run_loop
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations, make_base_test_augmentations

# %% misc setup

models_path = Path("models")
assert models_path.is_dir()

device = torch.device("cuda")
dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %% model setup

# ---- unet ----

# from scripts.models.unet_res import UNetRes
#
# model = UNetRes(len(dataset_stats["y_counts"]), "resnet34").to(device)
# model_name = "unet"
# num_epochs = 100
# batch_size = 16
# lr_min = 1e-6
#
# optimizer = torch.optim.AdamW([
#     {"params": model.get_encoder_parameters(), "lr": 1e-5},
#     {"params": model.get_decoder_parameters(), "lr": 1e-4},
#     {"params": model.get_head_parameters(), "lr": 1e-4},
# ])

# ---- deeplab ----

# from scripts.models.deeplab import DeepLabV3Plus
#
# model = DeepLabV3Plus(len(dataset_stats["y_counts"]), "resnet34").to(device)
# model_name = "deeplab"
# num_epochs = 100
# batch_size = 16
# lr_min = 1e-6
#
# optimizer = torch.optim.AdamW([
#     {"params": model.get_encoder_parameters(), "lr": 1e-5},
#     {"params": model.get_decoder_parameters(), "lr": 1e-4},
#     {"params": model.get_head_parameters(), "lr": 1e-4},
# ])

# ---- segformer ----

# from scripts.models.segformer import SegFormer
#
# model = SegFormer(len(dataset_stats["y_counts"]), variant="b2").to(device)
# model_name = "deeplab"
# num_epochs = 100
# batch_size = 4
# lr_min = 1e-6
#
# optimizer = torch.optim.AdamW([
#     {"params": model.get_encoder_parameters(), "lr": 1e-5},
#     {"params": model.get_decoder_parameters(), "lr": 1e-4},
# ])

# ---- transunet ----

# from scripts.models.transunet import TransUNet
#
# model = TransUNet(len(dataset_stats["y_counts"]), variant="b16").to(device)
# model_name = "transunet"
# num_epochs = 100
# batch_size = 2
# lr_min = 1e-6
#
# optimizer = torch.optim.AdamW([
#     {"params": model.get_encoder_parameters(), "lr": 1e-5},
#     {"params": model.get_decoder_parameters(), "lr": 1e-4},
#     {"params": model.get_head_parameters(), "lr": 1e-4},
# ])

# ---- swin ----

from scripts.models.swin_fpn import SwinFPN

model = SwinFPN(len(dataset_stats["y_counts"]), variant="b").to(device)
model_name = "swin"
num_epochs = 100
batch_size = 4
lr_min = 1e-7

optimizer = torch.optim.AdamW([
    {"params": model.get_encoder_parameters(), "lr": 1e-6},
    {"params": model.get_decoder_parameters(), "lr": 1e-5},
    {"params": model.get_head_parameters(), "lr": 1e-5},
])

# %% run

dataset_train, dataset_valid = DatasetTreeAI("ID", "train", 0) \
    .split(0.2, make_base_train_augmentations(), make_base_test_augmentations(), 42)

run_loop(
    device, model, optimizer,
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min),
    FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.7).to(device),
    num_epochs,
    dataset_train.loader(batch_size),
    dataset_valid.loader(batch_size),
    models_path / f"{model_name}_{time.strftime("%Y%m%d_%H%M%S")}",
    len(dataset_stats["y_counts"])
)
