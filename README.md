# Robust tree species identification using high-resolution RGB imagery and deep learning

Kai Kitagawa-Jones

## Goal / research questions

The overall goal of this project is to create tree genera semantic segmentation models on the TreeAI dataset. In
particular, the comparison between _in-distribution_ (ID) and _out-of-distribution_ (OOD) performance is explored.

The following are the main research questions:

1. Among CNN, ViT, and CNN-ViT hybrid architectures, which achieve the best ID performance for TS semantic segmentation?
2. Among CNN, ViT, and CNN-ViT hybrid architectures, which generalize best to OOD data, both with and without
   fine-tuning on a subset of the OOD data?
3. For both ID and OOD settings, which additional data sources, if any, lead to the greatest performance gains?

In terms of additional data sources, the following will be of focus:

- Climate data: using WorldClim bioclimatic variables
- Soil data: using Harmonized World Soil Database v2.0

## Related repositories

- [`kkitagawa/thesis`](https://gitlab.ethz.ch/kkitagawa/tree_segmentation): this repository, contains the core code for
  the
  thesis, mostly model definitions and train/eval scripts
- [`kkitagawa/treeai_segmentation`](https://gitlab.ethz.ch/kkitagawa/treeai_segmentation): code related to creating a
  semantic
  segmentation version of the TreeAI dataset (eventually to be integrated into [
  `ites-form/form-euler-dl`](https://gitlab.ethz.ch/ites-form/form-euler-dl))
- [`kkitagawa/treeai_labels`](https://gitlab.ethz.ch/kkitagawa/treeai_labels): code related to parsing and creting a
  phylogenetically organized/consistent summary of TreeAI labels
- [`kkitagawa/clim_soil_encoder`](https://gitlab.ethz.ch/kkitagawa/clim_soil_encoder): code related to creating and
  pre-training
  climate and soil encoders

## Code structure

Last update: 2026-07-06

```
thesis/                         ║
├─ data/                        ║ any intermediate data (this excludes the main TreeAI data, or anything really big)
├─ dependencies/                ║ various external dependencies
├─ figures/                     ║ output for figures
├─ models/                      ║ output for models
│  ├─ ${MODEL_NAME}_${DATATIME} ║   one folder per model
│  │  ├─ model.json             ║      model config
│  │  ├─ model.pt               ║      model weights
│  │  ├─ history.csv            ║      training history (including: loss, lr, confusion matrices, ...)
│  │  ├─ loss.png               ║      training loss plot
│  │  ├─ metrics_${DIST}.csv    ║      metrics for each target distribution
│  │  ├─ results_${DIST}.png    ║      results for each target distribution
│  │  └─ ...                    ║ (other misc files)
│  └─ ...                       ║ (more models)
├─ scripts/                     ║ scripts for processing data and training/evaluating models
├─ tasks/                       ║ for queing up batches of tasks to run overnight / on Euler
├─ run_tasks.sh                 ║ iterate through each script in tasks/ and run it
└─ ...                          ║ (other stuff)
```
