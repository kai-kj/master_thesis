# %%

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%

# unet w/o replay
# out_name = "unet_ftv_20260505_023132_without_replay"
# model_paths = {
#     "id": Path("models/unet_ftv_20260505_023132"),
#     "china": Path("models/unet_ftv_20260505_023132_china_20260505_100645"),
#     "czechia": Path("models/unet_ftv_20260505_023132_czechia_20260505_100757"),
#     "iran": Path("models/unet_ftv_20260505_023132_iran_20260505_100052"),
#     "mexico": Path("models/unet_ftv_20260505_023132_mexico_20260505_101809"),
#     "russia": Path("models/unet_ftv_20260505_023132_russia_20260505_105416"),
#     "south_africa": Path("models/unet_ftv_20260505_023132_south_africa_20260505_103450"),
# }

# unet w/ replay
# out_name = "unet_ftv_20260505_023132_with_replay"
# model_paths = {
#     "id": Path("models/unet_ftv_20260505_023132"),
#     "china": Path("models/unet_ftv_20260505_023132_china_20260505_133841"),
#     "czechia": Path("models/unet_ftv_20260505_023132_czechia_20260505_130023"),
#     "iran": Path("models/unet_ftv_20260505_023132_iran_20260505_133957"),
#     "mexico": Path("models/unet_ftv_20260505_023132_mexico_20260505_134553"),
#     "russia": Path("models/unet_ftv_20260505_023132_russia_20260505_134926"),
#     "south_africa": Path("models/unet_ftv_20260505_023132_south_africa_20260505_135708"),
# }

# segformer w/o replay
# out_name = "segformer_20260506_121655_without_replay"
# model_paths = {
#     "id": Path("models/segformer_20260506_121655"),
#     "china": Path("models/segformer_20260506_121655_china_20260515_201515"),
#     "czechia": Path("models/segformer_20260506_121655_czechia_20260515_201608"),
#     "iran": Path("models/segformer_20260506_121655_iran_20260515_202218"),
#     "mexico": Path("models/segformer_20260506_121655_mexico_20260515_202508"),
#     "russia": Path("models/segformer_20260506_121655_russia_20260515_202653"),
#     "south_africa": Path("models/segformer_20260506_121655_south_africa_20260515_203020"),
# }

# segformer w/ replay
# out_name = "segformer_20260506_121655_with_replay"
# model_paths = {
#     "id": Path("models/segformer_20260506_121655"),
#     "china": Path("models/segformer_20260506_121655_china_20260515_181757"),
#     "czechia": Path("models/segformer_20260506_121655_czechia_20260515_181913"),
#     "iran": Path("models/segformer_20260506_121655_iran_20260515_183035"),
#     "mexico": Path("models/segformer_20260506_121655_mexico_20260515_183538"),
#     "russia": Path("models/segformer_20260506_121655_russia_20260515_183835"),
#     "south_africa": Path("models/segformer_20260506_121655_south_africa_20260515_184452"),
# }

# transunet w/ replay
# out_name = "transunet_20260527_181604_with_replay"
# model_paths = {
#     "id": Path("models/transunet_20260527_181604"),
#     "china": Path("models/transunet_20260527_181604_china_20260529_125823"),
#     "czechia": Path("models/transunet_20260527_181604_czechia_20260529_130013"),
#     "iran": Path("models/transunet_20260527_181604_iran_20260529_131858"),
#     "mexico": Path("models/transunet_20260527_181604_mexico_20260529_132655"),
#     "russia": Path("models/transunet_20260527_181604_russia_20260529_133130"),
#     "south_africa": Path("models/transunet_20260527_181604_south_africa_20260529_134122"),
# }

# swin w/ replay
# out_name = "swin_fpn_20260529_145014_with_replay"
# model_paths = {
#     "id": Path("models/swin_fpn_20260529_145014"),
#     "china": Path("models/swin_fpn_20260529_145014_china_20260529_173831"),
#     "czechia": Path("models/swin_fpn_20260529_145014_czechia_20260529_173955"),
#     "iran": Path("models/swin_fpn_20260529_145014_iran_20260529_175350"),
#     "mexico": Path("models/swin_fpn_20260529_145014_mexico_20260529_175952"),
#     "russia": Path("models/swin_fpn_20260529_145014_russia_20260529_180330"),
#     "south_africa": Path("models/swin_fpn_20260529_145014_south_africa_20260529_181057"),
# }

out_name = "segformer_20260531_104250"
model_paths = {
    "id": Path("models/segformer_20260531_104250"),
    "china": Path("models/segformer_20260531_104250_china_20260531_164020"),
    "czechia": Path("models/segformer_20260531_104250_czechia_20260531_164110"),
    "iran": Path("models/segformer_20260531_104250_iran_20260531_164717"),
    "mexico": Path("models/segformer_20260531_104250_mexico_20260531_165002"),
    "russia": Path("models/segformer_20260531_104250_russia_20260531_165142"),
    "south_africa": Path("models/segformer_20260531_104250_south_africa_20260531_165502"),
}

model_names = {
    "id": "core model",
    "china": "china model",
    "czechia": "czechia model",
    "iran": "iran model",
    "mexico": "mexico model",
    "russia": "russia model",
    "south_africa": "south_africa model",
}

data_names = {
    "id": "core data",
    "china": "china data",
    "czechia": "czechia data",
    "iran": "iran data",
    "mexico": "mexico data",
    "russia": "russia data",
    "south_africa": "south_africa data",
}

for model_path in model_paths.values():
    assert model_path.is_dir()

# %%

metric = "f1"
agg = "weighted"

metrics = dict()

columns = list(model_paths.keys())

for model_id, model_path in model_paths.items():
    model_metrics = dict()

    for data_id in model_paths.keys():
        metrics_path = model_path / f"metrics_{data_id}.csv"

        if not metrics_path.is_file():
            continue

        df = pd.read_csv(metrics_path).set_index("metric")
        model_metrics[data_names[data_id]] = df.loc[agg, metric]

    metrics[model_names[model_id]] = model_metrics

metrics = pd.DataFrame(metrics).transpose()

# %%

xs = []
ys = []

for data_name in metrics.keys():
    for model_name in metrics.index:
        value: np.float64 = metrics.loc[model_name, data_name]  # type: ignore

        if not np.isnan(value):
            xs.append(f"{model_name} on {data_name}")
            ys.append(value)

    xs.append("")
    ys.append(0.0)

xs = xs[:-1]
ys = ys[:-1]

fig, ax = plt.subplots(figsize=(8, 5))

for model_name in model_names.values():
    ax.bar(
        [i for i in range(len(xs)) if xs[i].startswith(model_name)],
        [ys[i] for i in range(len(xs)) if xs[i].startswith(model_name)],
        align="center", label=model_name, hatch=".." if model_name == "core model" else "",
    )

    for i in [i for i in range(len(xs)) if xs[i].startswith(model_name)]:
        ax.text(i - 0.45, (int(ys[i] * 100) + 1) / 100 + 0.01, f"{ys[i]:.02f}", size="small")

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 1.15)
ax.set_ylabel(f"{agg} {metric} score")
ax.set_xticks([2.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22], data_names.values(), rotation=30)
ax.legend(loc="upper right", ncols=4, frameon=False)

fig.tight_layout()
fig.savefig(f"figures/tune_summary_{out_name}_{agg}_{metric}.png")
fig.show()
