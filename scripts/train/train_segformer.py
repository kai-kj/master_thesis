# %%

from pathlib import Path
import time
import torch
import json

from scripts.train.loss import FocalTverskyLoss
from scripts.train.train import run_loop
from scripts.models.segformer import SegFormer
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations, make_base_test_augmentations

# %%

models_path = Path("models")
assert models_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

dataset_train, dataset_valid = DatasetTreeAI("ID", "train", 0).split(
    0.2,
    train_augmentations=make_base_train_augmentations((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    test_augmentations=make_base_test_augmentations((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    random_state=42,
)

batch_size = 4
dataloader_train = dataset_train.loader(batch_size)
dataloader_valid = dataset_valid.loader(batch_size)

# %%

device = torch.device("cuda")

base_lr_encoder = 1e-5
base_lr_decoder = 1e-4
num_epochs = 50

model = SegFormer(len(dataset_stats["y_counts"]), variant="b2").to(device)
loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.3).to(device)

optimizer = torch.optim.AdamW([
    {"params": model.get_encoder_parameters(), "lr": base_lr_encoder},
    {"params": model.get_decoder_parameters(), "lr": base_lr_decoder},
])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=min(base_lr_encoder, base_lr_decoder) * 0.1
)

# %%

run_loop(
    device, model, optimizer, scheduler, loss_fn,
    num_epochs, dataloader_train, dataloader_valid,
    models_path / f"segformer_{time.strftime("%Y%m%d_%H%M%S")}",
    len(dataset_stats["y_counts"])
)
