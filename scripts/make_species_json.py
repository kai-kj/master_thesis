# %% imports

import pandas as pd
import graphviz
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

# %% setup

species_table_path = Path("data/intermediate/species.csv")
assert species_table_path.exists()

species_json_path = Path("data/intermediate/species.json")

# %% load data

species_table = pd.read_csv(species_table_path)

# %%

name_columns = [c for c in species_table.columns if c.endswith("_name")]
id_columns = [c for c in species_table.columns if c.endswith("_id")]

name_columns.reverse()
id_columns.reverse()

species_json = dict()

for _, row in species_table.iterrows():
    parent = species_json

    for (name_col, id_col) in zip(name_columns, id_columns):
        name_val: str = row[name_col]
        id_val: str = row[id_col]

        if not pd.isnull(name_val) and not pd.isnull(id_val):
            if id_val not in parent:
                parent[id_val] = {
                    "name": name_val,
                    "type": name_col.replace("_name", ""),
                    "count": None,
                    "children": dict(),
                }

            parent = parent[id_val]["children"]

    parent[row['id']] = {
        "name": row["name"],
        "type": "treeai",
        "count": row["count"],
        "children": dict(),
    }


def cumulative_count(node):
    if node["count"] is not None:
        return node["count"]

    node["count"] = 0

    for _, child in node["children"].items():
        node["count"] += cumulative_count(child)

    return node["count"]


cumulative_count(species_json["P"])

# %%

species_json_path.parent.mkdir(parents=True, exist_ok=True)
species_json_path.write_text(json.dumps(species_json, indent=4))
