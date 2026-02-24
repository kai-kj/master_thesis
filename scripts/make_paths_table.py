# %%

from pathlib import Path
import pandas as pd

# %%

data_path = Path("data/treeai")
assert data_path.exists()

out_path = Path("data/treeai/paths.csv")

# %%

paths = []

for label_type in "full", "part":
    data_path_sub = data_path / f"{label_type}_label"
    assert data_path_sub.exists()

    for image_dat_path in (data_path_sub / "images").glob("*.png"):
        image_loc_path = image_dat_path.with_suffix(".pgw")
        image_aux_path = image_dat_path.with_suffix(".png.aux.xml")
        image_lab_path = data_path_sub / "labels" / image_dat_path.name

        if not image_loc_path.exists():
            image_loc_path = image_loc_path.with_suffix(".wld")

        if not image_dat_path.exists():
            print(f"WARN: {image_dat_path} does not exist")
            continue

        if not image_loc_path.exists():
            print(f"WARN: {image_loc_path} does not exist")
            continue

        if not image_aux_path.exists():
            print(f"WARN: {image_aux_path} does not exist")
            continue

        if not image_lab_path.exists():
            print(f"WARN: {image_lab_path} does not exist")
            continue

        paths.append((
            str(label_type),
            str(image_dat_path),
            str(image_lab_path),
            str(image_loc_path),
            str(image_aux_path),
        ))

paths = pd.DataFrame(paths, columns=["label_type", "x", "y", "world", "meta"])

# %%

out_path.parent.mkdir(parents=True, exist_ok=True)
paths.to_csv(out_path, index=False)
