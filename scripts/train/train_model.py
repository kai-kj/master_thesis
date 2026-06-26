# %%

from pathlib import Path
import time
import torch
import json

from scripts.models.fusion import FusionModule
from scripts.train.loss import FocalTverskyLoss
from scripts.train.train import run_loop
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations_strong, make_base_test_augmentations

# %% misc setup

models_path = Path("models")
assert models_path.is_dir()

device = torch.device("cuda")

# %% fusion setup

from dependencies.clim_soil_encoder.model import MLPModel

use_clim = True
use_soil = True

fusion_model_path = {
    (True, False): Path("models/mlp_clim_20260619_131123"),
    (False, True): Path("models/mlp_soil_20260619_131725"),
    (True, True): Path("models/mlp_clim_soil_20260619_132336"),
}[(use_clim, use_soil)]

fusion_model = MLPModel.load(fusion_model_path / "model.json", fusion_model_path / "model.pt")
fusion_model.set_apply_softmax(False)

# ---- none ----

# fusion_mode = None
# fusion_encoder = None

# ---- concat ----

# fusion_mode = "concat"
# fusion_encoder = fusion_model.encoder.to(device)

# ---- film ----

fusion_mode = "film"
fusion_encoder = fusion_model.encoder.to(device)

# ---- logit ----

# fusion_mode = "logit"
# fusion_encoder = fusion_model.to(device)

# %% model setup

# ---- unet ----

from scripts.models.unet_res import UNetRes

model = UNetRes(10, "resnet34", fusion_mode=fusion_mode, fusion_encoder=fusion_encoder).to(device)
model_name = "unet"
num_epochs = 100
batch_size = 16
lr_min = 1e-6

optimizer_params = [
    {"params": model.get_encoder_parameters(), "lr": 1e-5},
    {"params": model.get_decoder_parameters(), "lr": 1e-4},
    {"params": model.get_head_parameters(), "lr": 1e-4},
]

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

# from scripts.models.swin_fpn import SwinFPN
#
# model = SwinFPN(len(dataset_stats["y_counts"]), variant="b").to(device)
# model_name = "swin"
# num_epochs = 100
# batch_size = 4
# lr_min = 1e-7
#
# optimizer = torch.optim.AdamW([
#     {"params": model.get_encoder_parameters(), "lr": 1e-6},
#     {"params": model.get_decoder_parameters(), "lr": 1e-5},
#     {"params": model.get_head_parameters(), "lr": 1e-5},
# ])

# %% run

if fusion_mode is not None:
    optimizer_params += [
        {"params": model.get_fusion_parameters(), "lr": 1e-4},
        {"params": model.get_fusion_encoder_parameters(), "lr": 1e-5},
    ]

    model_name += f"_{fusion_mode}_fusion" + "_clim" * use_clim + "_soil" * use_soil

dataset_train, dataset_valid = DatasetTreeAI("ID", "train", 0, return_clim=use_clim, return_soil=use_soil) \
    .split(0.2, make_base_train_augmentations_strong(), make_base_test_augmentations(), 42)

optimizer = torch.optim.AdamW(optimizer_params)

run_loop(
    device, model, optimizer,
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min),
    FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.7).to(device),
    num_epochs,
    dataset_train.loader(batch_size),
    dataset_valid.loader(batch_size),
    models_path / f"{model_name}_{time.strftime("%Y%m%d_%H%M%S")}",
    10
)
