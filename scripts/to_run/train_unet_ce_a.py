# %%

from pathlib import Path
import time
import torch
import json
import albumentations as A

from scripts.train.loss import calc_class_weights
from scripts.train.train import run_loop
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI, make_base_train_augmentations, make_base_test_augmentations

# %%

models_path = Path("models")
assert models_path.is_dir()

dataset_stats = json.loads(Path("data/dataset_stats.json").read_text())

# %%

dataset_train, dataset_valid = DatasetTreeAI("ID", "train", 0).split(
    0.2,
    train_augmentations=A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-90.0, 90.0), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.5),
        A.HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-5, 5), val_shift_limit=(-5, 5), p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), p=0.125),
        A.Normalize(mean=dataset_stats["x_mean"], std=dataset_stats["x_std"], max_pixel_value=1.0, p=1.0),
    ]),
    test_augmentations=make_base_test_augmentations(dataset_stats["x_mean"], dataset_stats["x_std"]),
    random_state=42,
)

batch_size = 6
dataloader_train = dataset_train.loader(batch_size)
dataloader_valid = dataset_valid.loader(batch_size)

# %%

device = torch.device("cuda")

base_lr = 1e-3
num_epochs = 100

model = UNet(len(dataset_stats["y_counts"]), norm="instance").to(device)
weights = calc_class_weights(torch.Tensor(list(dataset_stats["y_counts"].values()))).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weights)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr * 0.01)

# %%

run_loop(
    device, model, optimizer, scheduler, loss_fn,
    num_epochs, dataloader_train, dataloader_valid,
    models_path / time.strftime("%Y%m%d_%H%M%S")
)
