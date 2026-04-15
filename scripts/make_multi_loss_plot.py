# %%

from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd

# %%

# model_path = Path("models/20260414_100556")
model_path = Path("models/20260402_185900")
assert model_path.is_dir()


# %%

def get_parents(path: Path) -> list[tuple[Path | None, float]]:
    parent = None
    lr = None

    for line in (path / "train.py.txt").read_text().split("\n"):
        line = line.strip()

        if line.startswith("#"):
            continue

        if match := re.search(r"models_path\s*/\s*\"(\w*)\"", line):
            parent = path.parent / Path(match.group(1))

        if match := re.search(r"lr\s*=\s*([\w-]*)", line):
            lr = float(match.group(1))

    assert lr is not None

    return [(parent, lr), *get_parents(parent)] if parent else [(None, lr)]


def get_model_hierarchy(path: Path) -> list[tuple[str, float, list[float], list[float]]]:
    model_parents: list[tuple[Path | None, float | None]] = [(path, None), *get_parents(path)]

    return list(reversed([  # type: ignore
        (
            path.name, lr,
            pd.read_csv(path / "loss.csv")["train_loss"].to_list(),
            pd.read_csv(path / "loss.csv")["valid_loss"].to_list(),
        )
        for path, lr
        in zip(list(zip(*model_parents))[0][:-1], list(zip(*model_parents))[1][1:])
    ]))


model_hierarchy = get_model_hierarchy(model_path)

print("model hierarchy:")

for h in model_hierarchy:
    print(f"- {h[0]}")

# %%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

p1, = ax1.plot(
    [l for h in model_hierarchy for l in h[2]],
    color=color_cycle[0], label="train loss"
)

p2, = ax1.plot(
    [l for h in model_hierarchy for l in h[3]],
    color=color_cycle[1], label="valid loss"
)

p3, = ax2.plot(
    [l for h in model_hierarchy for l in [h[1]] * len(h[2])],
    color=color_cycle[2], linestyle="--", label="learning rate"
)

ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.spines[["top"]].set_visible(False)
ax1.legend(handles=[p1, p2, p3], loc="upper right", )

ax2_ylim_margin = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.25
ax2.set_ylim((ax2.get_ylim()[0] - ax2_ylim_margin, ax2.get_ylim()[1] + ax2_ylim_margin))
ax2.set_ylabel("learning rate")
ax2.spines[["top"]].set_visible(False)

fig.tight_layout()
fig.savefig(model_path / "loss_full.png")
fig.show()
