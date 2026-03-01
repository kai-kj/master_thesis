# %% imports

import warnings
from pathlib import Path
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

# %% setup

plantae_dwca_path = Path("data/catalogueoflife/plantae_coldp.tsv")
assert plantae_dwca_path.exists()

shapefile_path = Path("data/treeai_shape/shapes_v4/202601_FullyL_103402.shp")
assert shapefile_path.exists()

species_path = Path("data/intermediate/species.csv")

# %% load data - catalogueoflife

plantae = pd.read_csv(plantae_dwca_path, sep="\t") \
    .drop(columns=["col:authorship"]) \
    .rename(columns={
    "col:ID": "id",
    "col:parentID": "parent_id",
    "col:status": "status",
    "col:rank": "rank",
    "col:scientificName": "name"
})

# %% load data - trees

trees = gpd.read_file(shapefile_path)[["speciesLat"]]
trees = trees.rename(columns={"speciesLat": "name"})

# %%

species_a = pd.DataFrame(trees.groupby(["name"]).size(), columns=["count"]).reset_index().reset_index()
species_a = species_a.rename(columns={"index": "treeai_id", "name": "treeai_name", "count": "treeai_count"})

species_a.insert(
    2,
    "treeai_name_clean",
    species_a["treeai_name"].map(lambda s: s.lower().replace(" sp.", "").strip())
)

manual_override = {
    "pinus mugo prostrata": "pinus mugo",
    "na": "unknown",
}

species_a["treeai_name_clean"] = species_a["treeai_name_clean"].map(lambda v: manual_override.get(v, v))

print(f"found {len(species_a)} species")

# %% find matches

species_b = None

for _, s in species_a.iterrows():
    matches = plantae[plantae["name"].str.contains(s["treeai_name_clean"], case=False)]

    matches.insert(0, "treeai_id", s["treeai_id"])
    matches.insert(1, "treeai_name_clean", s["treeai_name_clean"])

    if species_b is None:
        species_b = matches
    else:
        species_b = pd.concat([species_b, matches])

species_b = species_b.reset_index(drop=True)  # type: ignore

# %% score matches

status_score = ["accepted", "provisionally accepted", "synonym", "ambiguous synonym", "misapplied"]

species_c = species_b.copy()
species_c["match_score"] = species_c["name"].str.len() - species_c["treeai_name_clean"].str.len()
species_c["status_score"] = species_c["status"].map(lambda v: status_score.index(v))

# %% filter matches

species_d = species_c \
    .sort_values(["treeai_id", "match_score", "status_score"]) \
    .groupby(["treeai_id"]) \
    .head(1)[["treeai_id", "id"]]

print(f"species not found:")

for _, s in species_a.iterrows():
    if s["treeai_id"] not in species_d["treeai_id"].tolist():
        print(f"  {s['treeai_id']} \"{s['treeai_name']}\"")


# %%

def get_parent(by: str, target: str, col_id: str) -> str | None:
    assert by in plantae.columns

    child_df = plantae[plantae["id"] == col_id]
    child = None if child_df.empty else child_df.iloc[0]

    if child is None:
        return None

    if child[by] == target:
        return child["id"]

    parent_df = plantae[plantae["id"] == child["parent_id"]]
    parent = None if parent_df.empty else parent_df.iloc[0]

    if parent is None:
        return None

    if parent[by] != target:
        return get_parent(by, target, parent["id"])
    else:
        return parent["id"]


def get_full(t: pd.DataFrame) -> pd.DataFrame:
    return t.merge(plantae, on="id", how="left")


# %% remove synonyms

species_e = species_d.copy()
species_e["id"] = species_e["id"].map(lambda i: get_parent("status", "accepted", i))

species_e_full = get_full(species_e)
assert sum(species_e_full["status"] != "accepted") == 0

# %% get species info


print("finding species...")
species_f_species = species_e.copy()
species_f_species["id"] = species_f_species["id"].map(lambda i: get_parent("rank", "species", i))

print("finding genus...")
species_f_genus = species_e.copy()
species_f_genus["id"] = species_f_genus["id"].map(lambda i: get_parent("rank", "genus", i))

print("finding family...")
species_f_family = species_e.copy()
species_f_family["id"] = species_f_family["id"].map(lambda i: get_parent("rank", "family", i))

print("finding order...")
species_f_order = species_e.copy()
species_f_order["id"] = species_f_order["id"].map(lambda i: get_parent("rank", "order", i))

print("finding class...")
species_f_class = species_e.copy()
species_f_class["id"] = species_f_class["id"].map(lambda i: get_parent("rank", "class", i))

print("finding phylum...")
species_f_phylum = species_e.copy()
species_f_phylum["id"] = species_f_phylum["id"].map(lambda i: get_parent("rank", "phylum", i))

print("finding kingdom...")
species_f_kingdom = species_e.copy()
species_f_kingdom["id"] = species_f_kingdom["id"].map(lambda i: get_parent("rank", "kingdom", i))

# %%

species_f_species_full = get_full(species_f_species)[["treeai_id", "id", "name"]] \
    .rename(columns={"id": "species_id", "name": "species_name"})

species_f_genus_full = get_full(species_f_genus)[["treeai_id", "id", "name"]] \
    .rename(columns={"id": "genus_id", "name": "genus_name"})

species_f_family_full = get_full(species_f_family)[["treeai_id", "id", "name"]] \
    .rename(columns={"id": "family_id", "name": "family_name"})

species_f_order_full = get_full(species_f_order)[["treeai_id", "id", "name"]] \
    .rename(columns={"id": "order_id", "name": "order_name"})

species_f_class_full = get_full(species_f_class)[["treeai_id", "id", "name"]] \
    .rename(columns={"id": "class_id", "name": "class_name"})

species_f_phylum_full = get_full(species_f_phylum)[["treeai_id", "id", "name"]] \
    .rename(columns={"id": "phylum_id", "name": "phylum_name"})

species_f_kingdom_full = get_full(species_f_kingdom)[["treeai_id", "id", "name"]] \
    .rename(columns={"id": "kingdom_id", "name": "kingdom_name"})

species_g = species_a \
    .merge(species_f_species_full, on="treeai_id", how="left") \
    .merge(species_f_genus_full, on="treeai_id", how="left") \
    .merge(species_f_family_full, on="treeai_id", how="left") \
    .merge(species_f_order_full, on="treeai_id", how="left") \
    .merge(species_f_class_full, on="treeai_id", how="left") \
    .merge(species_f_phylum_full, on="treeai_id", how="left") \
    .merge(species_f_kingdom_full, on="treeai_id", how="left")

species_g = species_g.rename(columns={
    "treeai_id": "id", "treeai_name": "name", "treeai_count": "count"
}).drop(columns=["treeai_name_clean"])

# %%

species_path.parent.mkdir(parents=True, exist_ok=True)
species_g.to_csv(species_path, index=False)
