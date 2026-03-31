# %%

from pathlib import Path
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision.transforms.v2 as tv_transforms
import sklearn

from scripts.check_input_outputs import check_input_outputs
from scripts.util.train import run_epoch
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI
from scripts.util.seed import seed

# %%

inputs: dict[str, Path] = {
    "patches": Path("data/intermediate/patches_b.gpkg"),
    "labels": Path("data/intermediate/labels.csv"),
    # "base_model": Path("models/20260330_140518"),
}

outputs: dict[str, Path] = {
    "models": Path("models"),
}

check_input_outputs(inputs, outputs)  # type: ignore

device = torch.device("cuda")
batch_size = 6

seed(42)

# %%

train_transform = tv_transforms.Compose([
    tv_transforms.RandomHorizontalFlip(),
    tv_transforms.RandomVerticalFlip(),
    tv_transforms.RandomRotation([0, 360]),
    tv_transforms.RandomCrop([512, 512]),
])

valid_transform = tv_transforms.Compose([
    tv_transforms.CenterCrop([512, 512]),
])

dataset_all = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "both")
train_idx, valid_idx = sklearn.model_selection.train_test_split(range(len(dataset_all)), test_size=0.25)

dataset_train = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "both", train_idx, train_transform)
dataset_valid = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "both", valid_idx, valid_transform)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=6, prefetch_factor=2, persistent_workers=True
)

dataloader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=batch_size, shuffle=True,
    pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True
)

# %%

class_counts = np.array(list(dataset_all.class_counts.values()))
class_freq = class_counts / np.sum(class_counts)

weights = np.log(1 / class_freq)
weights = weights / weights.sum() * weights.shape[0]

model = UNet(dataset_all.n_classes).to(device)

if "base_model" in inputs:
    model.load_state_dict(torch.load(inputs["base_model"] / "model.pt"))

num_epochs = 100

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2 * num_epochs)

loss_fn = torch.nn.CrossEntropyLoss(torch.Tensor(weights).to(device))
scaler = torch.amp.grad_scaler.GradScaler(device=device.type)

# %%

model_path = outputs["models"] / time.strftime("%Y%m%d_%H%M%S")
model_path.mkdir(parents=True, exist_ok=False)

(model_path / "class.txt").write_text(str(type(model)))

train_loss = []
valid_loss = []

start_time = time.time()

for epoch in range(num_epochs):
    print(f"epoch {epoch + 1}/{num_epochs}")

    curr_train_loss, curr_valid_loss = run_epoch(
        device,
        model,
        optimizer, loss_fn,
        dataloader_train, dataloader_valid,
        scaler
    )

    scheduler.step()

    train_loss.append(curr_train_loss)
    valid_loss.append(curr_valid_loss)

    print(f"  train loss: {train_loss[-1]:.4f}\n  valid loss: {valid_loss[-1]:.4f}")

    if len(valid_loss) == 1 or valid_loss[-1] < np.min(valid_loss[:-1]):
        print("  best model, saving")
        torch.save(model.state_dict(), model_path / "model.pt")

    if (epoch + 1) % 10 == 0:
        fig, ax = plt.subplots()

        ax.plot(train_loss, label="train loss")
        ax.plot(valid_loss, label="valid loss")

        ax.set_xlim(0, num_epochs)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(model_path / "loss.png")
        fig.show()

end_time = time.time()

loss = pd.DataFrame(
    zip(list(i + 1 for i in range(len(train_loss))), train_loss, valid_loss),
    columns=["epoch", "train_loss", "valid_loss"]
)

loss.to_csv(model_path / "loss.csv", index=False)

duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

print(f"done, training time for {num_epochs} epochs: {duration}")
