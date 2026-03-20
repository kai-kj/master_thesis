# Kai's thesis project

## Scripts description

Last update: 2026-03-12

In/out describe what non-raw data is required and produced by each script. In general, each non-raw data is immutable (
once it's created, other scripts don't modify them). Raw data that is required by the scripts is not included.

### ```check_rasters_exist.py```

Check that all rasters mentioned in the raw TreeAI shapefile exist.

- in: none
- out: none

### ```make_patches.py```

Create image patches for a given patch size and patch overlap.

- in: `trees.gpkg`, `sites_a.gpkg`
- out: `rasters_merged/`, `patches/*x*_*/`, `patches_*x*_*.gpkg`

### ```make_site_summary.py```

Make a summary of the sites, in terms of species and genus distribution.

- in: `sites_a.gpkg`, `trees.gpkg`, `species.csv`
- out: none

### ```make_sites_table.py```

Make a table that groups trees into sites.

- in: `trees.gpkg`
- out: `sites_a.gpkg`

### ```make_species_json.py```

Make a json-based representation of the tree species in the dataset.

- in: `species.csv`
- out: `species.json`

### ```make_species_summary.py```

Make phylogenetic tree visualization of the relationship between the various tree species in the dataset.

- in: `species.json`
- out: none

### ```make_species_table.py```

Make a table that lists and describes all the species in the dataset.

- in: `data/treeai/shapes_v4/202601_FullyL_103402.shp`
- out: `species.csv`

### ```make_trees_table.py```

Process the raw shapefiles into a clean table that describes all trees in the dataset.

- in: `species.csv`
- out: `trees.gpkg`