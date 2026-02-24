# %%

from pathlib import Path
import pandas as pd
import re
import utm

# %%

paths_table_path = Path("data/treeai/paths.csv")
assert paths_table_path.exists()

out_path = Path("data/treeai/summary_a.csv")

# %%

paths = pd.read_csv(paths_table_path)

summary = []

for _, row in paths.iterrows():
    label_type = row["label_type"]
    x = row["x"]
    y = row["y"]
    world = Path(row["world"]).read_text().split("\n")
    meta = Path(row["meta"]).read_text()

    world_a = float(world[0])
    world_d = float(world[1])
    world_b = float(world[2])
    world_e = float(world[3])
    world_c = float(world[4])
    world_f = float(world[5])

    assert world_d == 0
    assert world_b == 0

    utm_zone = re.search(r"UTM zone (\w+)", meta).group(1)  # type: ignore

    utm_northern = utm_zone[-1] == "N"
    utm_zone = int(utm_zone[:-1])

    lat, lon = utm.to_latlon(world_c, world_f, utm_zone, northern=utm_northern, strict=False)

    summary.append((label_type, x, y, world_a, world_e, world_c, world_f, utm_zone, utm_northern, lat, lon))

summary = pd.DataFrame(
    summary,
    columns=[
        "label_type", "x", "y", "world_a", "world_e", "world_c", "world_f", "utm_zone", "utm_northern", "lat", "lon"
    ],
)

# %%

out_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_path, index=False)
