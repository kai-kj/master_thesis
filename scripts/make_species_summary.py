# %% imports

import pandas as pd
import graphviz
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# %% setup

species_table_path = Path("data/intermediate/species.csv")
assert species_table_path.exists()

out_path = Path("out")

# %% load data

species_table = pd.read_csv(species_table_path)

# %% make graph

ranks = [r.replace("_name", "") for r in species_table.columns if r.endswith("_name")]

nodes: dict[str, set[tuple[str, str]]] = dict()
edges: set[tuple[str, str]] = set()

for r in ranks:
    nodes[r] = set()

for _, row in species_table.iterrows():
    prev = row[f"{ranks[0]}_name"]

    first: None | tuple[str, str] = None

    for rank_curr, rank_next in zip(ranks[:-1], ranks[1:]):
        curr_label = row[f"{rank_curr}_name"]
        next_label = row[f"{rank_next}_name"]

        curr_label = curr_label if not pd.isnull(curr_label) else ""
        next_label = next_label if not pd.isnull(next_label) else ""

        curr_name = curr_label.lower().replace(" ", "_") + "_col" * (curr_label != "")
        next_name = next_label.lower().replace(" ", "_") + "_col" * (next_label != "")

        nodes[rank_curr].add((curr_name, curr_label))
        nodes[rank_next].add((next_name, next_label))
        edges.add((curr_name, next_name))

        if not first and curr_name != "":
            first = (rank_curr, curr_name)

    if first:
        label = f"{row['name']}"
        name = label.lower().replace(" ", "_") + "_treeai"
        label = f"{label}: {row['count']}"
        nodes[first[0]].add((name, label))
        edges.add((name, first[1]))

nodes = {k: {vv for vv in v if vv[1] != ""} for k, v in nodes.items()}
edges = {v for v in edges if v[0] != "" and v[1] != "" and v[0] != v[1]}

graph = graphviz.Graph()

for v in edges:
    graph.edge(v[0], v[1])

for k, v in nodes.items():
    graph.attr(rank="same")
    for vv in v:
        if vv[0].endswith("_treeai"):
            graph.node(
                vv[0], shape="none",
                label=f"<<FONT COLOR=\"red\"><B>{vv[1]}</B></FONT>>"
            )
        else:
            graph.node(vv[0], shape="none", label=vv[1])

graph.attr(layout="dot")
graph.attr(rankdir="RL")
graph.attr(ranksep="1")
graph.attr(nodesep="0")
graph.render(out_path / "species_summary_tree", format="png", cleanup=True)

# %%

species_histogram = species_table[["species_name", "count"]].groupby("species_name").sum().reset_index()

fig, ax = plt.subplots(figsize=(24, 6))

ax.bar(np.arange(len(species_histogram)), species_histogram["count"])

ax.spines[['right', 'top']].set_visible(False)
ax.set_xticks(np.arange(len(species_histogram)), labels=species_histogram["species_name"], rotation=90)
ax.set_ylabel("occurrences")
ax.set_title("Tree occurrences by species")
fig.tight_layout()

fig.savefig(out_path / "species_summary_species_hist.png")
fig.show()

# %%

genus_histogram = species_table[["genus_name", "count"]].groupby("genus_name").sum().reset_index()

fig, ax = plt.subplots(figsize=(16, 6))

ax.bar(np.arange(len(genus_histogram)), genus_histogram["count"])

ax.spines[['right', 'top']].set_visible(False)
ax.set_xticks(np.arange(len(genus_histogram)), labels=genus_histogram["genus_name"], rotation=90)
ax.set_ylabel("occurrences")
ax.set_title("Tree occurrences by genus")
fig.tight_layout()

fig.savefig(out_path / "species_summary_genus_hist.png")
fig.show()

# %%

family_histogram = species_table[["family_name", "count"]].groupby("family_name").sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(np.arange(len(family_histogram)), family_histogram["count"])

ax.spines[['right', 'top']].set_visible(False)
ax.set_xticks(np.arange(len(family_histogram)), labels=family_histogram["family_name"], rotation=90)
ax.set_ylabel("occurrences")
ax.set_title("Tree occurrences by family")
fig.tight_layout()

fig.savefig(out_path / "species_summary_family_hist.png")
fig.show()
