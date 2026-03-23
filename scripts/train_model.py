# %%

from pathlib import Path
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd

from scripts.check_input_outputs import check_input_outputs
from scripts.util.train import run_epoch
from scripts.models.unet import UNet
from scripts.dataset import DatasetTreeAI

# %%

inputs: dict[str, Path] = {
    "patches": Path("data/intermediate/patches_b.gpkg"),
    "labels": Path("data/intermediate/labels.csv"),
}

outputs: dict[str, Path] = {
    "models": Path("models"),
}

check_input_outputs(inputs, outputs)

device = torch.device("cuda")
batch_size = 6

# %%

generator = torch.Generator().manual_seed(42)

dataset_all = DatasetTreeAI(inputs["patches"], inputs["labels"], "ID", "train")
dataset_train, dataset_valid = torch.utils.data.random_split(dataset_all, [0.8, 0.2], generator)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=generator)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, generator=generator)

# %%

# class_counts = np.array(list(dataset_all.class_counts.values()))
# weights = np.sum(class_counts) / class_counts
# weights = weights / weights.sum() * weights.shape[0]

model = UNet(dataset_all.n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()  # https://github.com/Nacriema/Loss-Functions-For-Semantic-Segmentation

# %%

model_path = outputs["models"] / time.strftime("%Y%m%d_%H%M%S")
model_path.mkdir(parents=True, exist_ok=False)

(model_path / "class.txt").write_text(str(type(model)))

num_epochs = 5
train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    print(f"epoch {epoch + 1}/{num_epochs}")

    curr_train_loss, curr_valid_loss = run_epoch(
        device,
        model,
        optimizer, loss_fn,
        dataloader_train, dataloader_valid
    )

    torch.save(model.state_dict(), model_path / "model.pt")

    train_loss.append(curr_train_loss)
    valid_loss.append(curr_valid_loss)

    print(f"  train loss: {train_loss[-1]:.4f}\n  valid loss: {valid_loss[-1]:.4f}")

torch.cuda.empty_cache()

loss = pd.DataFrame(
    zip(list(i + 1 for i in range(len(train_loss))), train_loss, valid_loss),
    columns=["epoch", "train_loss", "valid_loss"]
)

loss.to_csv(model_path / "loss.csv", index=False)

print("done")

# %%

fig, ax = plt.subplots()

ax.plot(train_loss, label="train loss")
ax.plot(valid_loss, label="valid loss")
ax.hlines(
    [train_loss[-1], valid_loss[-1]], 0, len(train_loss),
    label="final loss", linestyles="dashed", color="red"
)

ax.text(len(train_loss), train_loss[-1], f"{train_loss[-1]:.3f}")
ax.text(len(valid_loss), valid_loss[-1], f"{valid_loss[-1]:.3f}")

ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(model_path / "loss.png")
fig.show()
