# %%

from pathlib import Path
import time
import torch
import json

from scripts.train.loss import calc_class_weights
from scripts.train.train import run_loop
from scripts.models.unet_scratch import UNet
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations, make_base_test_augmentations

# %%

model_path = Path("models/20260504_070305")
assert model_path.is_dir()

dist_name = "iran"

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

dataset_train, dataset_valid = DatasetTreeAI(f"OOD_{dist_name}", "train", 0).split(
    0.2,
    train_augmentations=make_base_train_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
    test_augmentations=make_base_test_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
    random_state=42,
)

batch_size = 6
dataloader_train = dataset_train.loader(batch_size)
dataloader_valid = dataset_valid.loader(batch_size)

# %%

device = torch.device("cuda")

base_lr = 5e-5
num_epochs = 25

model = UNet.load(model_path / "model.json", model_path / "model.pt").to(device)

# for module in model.get_encoder_modules():
#     for param in module.parameters():
#         param.requires_grad = False

weights = calc_class_weights(torch.Tensor(list(dataset_train.calc_stats()[2].values()))).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weights)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr * 0.01)

# %%

run_loop(
    device, model, optimizer, scheduler, loss_fn,
    num_epochs, dataloader_train, dataloader_valid,
    f"{model_path.as_posix()}_{dist_name}_{time.strftime("%Y%m%d_%H%M%S")}",
    len(dataset_stats["y_counts"])
)
