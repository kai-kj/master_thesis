# %% imports

import pandas as pd
import graphviz
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

# %% setup

species_path = Path("data/intermediate/species.json")
assert species_path.exists()

out_path = Path("out")

# %% load data

species = json.loads(species_path.read_text())

# %% make graph


nodes = {}
edges = []


def add_children(children, parent=None):
    for k, v in children.items():
        if parent: edges.append((parent, k))
        if v["type"] not in nodes: nodes[v["type"]] = []

        color = "red" if v["type"] == "treeai" else "black"
        nodes[v["type"]].append((
            k,
            f"<<FONT COLOR=\"{color}\"><B>{k}</B><BR/>{v['name']}<BR/>count: {v['count']}</FONT>>"
        ))

        if "children" in v: add_children(v["children"], k)


add_children(species)

graph = graphviz.Graph()

for e in edges:
    if e[0].startswith("TAI") or e[1].startswith("TAI"):
        graph.edge(e[0], e[1], style="dotted", penwidth="4")
    else:
        graph.edge(e[0], e[1])

for k, v in nodes.items():
    subgraph = graphviz.Graph()
    subgraph.attr(rank="same")

    for vv in v:
        subgraph.node(vv[0], shape="none", label=vv[1])

    graph.subgraph(subgraph)

# graph.attr(layout="twopi")
# graph.attr(root="plantae_col")
# graph.attr(overlap_scaling="5")
# graph.attr(ranksep="4")

graph.attr(layout="dot")
graph.attr(rankdir="LR")
graph.attr(ranksep="2")
graph.attr(nodesep="0")
graph.attr(splines="polyline")

graph.render(out_path / "species_summary_tree", format="png", cleanup=True)

# %%

pd.json_normalize(species, sep="_").to_dict(orient="records")[0]


# %%

def get_counts(group):
    records = pd.json_normalize(species, sep="_").to_dict(orient="records")[0]

    matches = [
        "_".join(k.split("_")[:-1])  # type: ignore
        for k, v in records.items()
        if k.endswith("_type") and v == group  # type: ignore
    ]

    ids = [m.split("_")[-1] for m in matches]
    names = [records[f"{k}_name"] for k in matches]
    counts = [records[f"{k}_count"] for k in matches]

    if sum(counts) != species["P"]["count"]:
        ids.append("(other)")
        names.append("(other)")
        counts.append(species["P"]["count"] - sum(counts))

    return ids, names, counts


# %%

for group, title in [
    ("treeai", "TreeAI label"), ("species", "COL species"),
    ("genus", "COL genus"), ("family", "COL family")
]:
    hist = get_counts(group)

    fig, ax = plt.subplots(figsize=(int(len(hist[0]) * 0.2), 4))
    ax.bar(np.arange(len(hist[2])), hist[2])

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks(np.arange(len(hist[2])), labels=hist[0], rotation=90)
    ax.set_ylabel("occurrences")
    ax.set_title(f"Occurrences by {title}")
    fig.tight_layout()

    fig.savefig(out_path / f"species_summary_{group}_hist.png")
    fig.show()
