# %%

from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd

# %%

model_path = Path("models/20260422_204630")
assert model_path.is_dir()

# %%

history: pd.DataFrame = None  # type: ignore

if (model_path / "loss.csv").exists():
    history = pd.read_csv(model_path / "loss.csv")
elif (model_path / "history.csv").exists():
    history = pd.read_csv(model_path / "history.csv")
else:
    raise RuntimeError(f"loss.csv or history.csv not found")

# %%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

p1, = ax1.plot(history["train_loss"], color=color_cycle[0], label="train loss")
p2, = ax1.plot(history["valid_loss"], color=color_cycle[1], label="valid loss")
p3, = ax2.plot(history["lr"], color=color_cycle[2], linestyle="--", label="learning rate")

ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.spines[["top"]].set_visible(False)
ax1.legend(handles=[p1, p2, p3], loc="upper right", )

ax2_ylim_margin = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.125
ax2.set_ylim(max(0, ax2.get_ylim()[0] - ax2_ylim_margin), ax2.get_ylim()[1] + ax2_ylim_margin)
ax2.set_ylabel("learning rate")
ax2.spines[["top"]].set_visible(False)

fig.tight_layout()
fig.savefig(model_path / "loss_full.png")
fig.show()
