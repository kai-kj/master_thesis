# %% imports

import pandas as pd
import graphviz
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json

from scripts.check_input_outputs import check_input_outputs

# %% setup


inputs: dict[str, Path] = {
    "species": Path("data/intermediate/species.json"),
}

outputs: dict[str, Path] = {
    "species_summary": Path("figures/species_summary"),
}

check_input_outputs(inputs, outputs)

# %% load data

species = json.loads(inputs["species"].read_text())

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

graph.attr(layout="dot")
graph.attr(rankdir="LR")
graph.attr(ranksep="2")
graph.attr(nodesep="0")
graph.attr(splines="polyline")

graph.render(outputs["species_summary"], format="png", cleanup=True)
