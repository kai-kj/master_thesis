# Robust tree species semantic segmentation across geographic regions using RGB imagery and environmental data fusion

## Description

This repository contains code for the main part of the thesis, in particular, the training and evaluation of the
semantic segmentation models.

## Structure

```
thesis/                     |
├─ data                     | small intermediate data, produced by the numbered scripts in scripts/
│ ├─ dataset_stats.json     | general stats used for standardization etc
│ ├─ labels_map.csv         | mapping from TreeAI labels to the smaller subset used in theis
│ ├─ labels_map_africa.csv  | same as above, but for S. Africa, which has a different set
│ └─ sites_split.gpkg       | train-test-val split version of the data
├─ dependencies/            | git submodules
│ ├─ clim_soil_encoder/     | -> git@gitlab.ethz.ch:kkitagawa/clim_soil_encoder.git
│ ├─ TransUNet/             | -> https://github.com/Beckschen/TransUNet.git
│ ├─ treeai_labels/         | -> git@gitlab.ethz.ch:kkitagawa/treeai_labels.git
│ └─ treeai_segmentation/   | -> git@gitlab.ethz.ch:kkitagawa/treeai_segmentation.git
├─ figures/                 | figures produced for the thesis
├─ tables/                  | latex tables produced for the thesis
├─ scripts/                 | the main scripts for the project, see below for details
├─ tasks/                   | for running tasks overnight / on Euler
├─ run_tasks.sh             | iterate through each script in tasks/ and run it
├─ workflow.pdf             | a extract from the bi-weekly presentation, about the general workflow
└─ ...                      |
```

## Models

|            | category    | variant   | pre-training | #params |
|------------|-------------|-----------|--------------|---------|
| UNet       | CNN         | ResNet-34 | ImageNet-1k  | 24.44M  | 
| DeepLabV3+ | CNN         | ResNet-34 | ImageNet-1k  | 22.44M  | 
| Segformer  | transformer | b2        | ImageNet-1k  | 27.35M  | 
| TransUNet  | hybrid      | b16       | ImageNet-21k | 92.31M  | 
| SwinFPN    | hybrid      | base      | ImageNet-1k  | 89.60M  | 

## Data splits/distributions

- Core data (mainly Schwarzwald and Quebec) $\leftarrow$ makes up most of the data in terms of volume
- OOD data
    - Austria $\leftarrow$ from the Austria dataset (not TreeAI)
    - China
    - Czechia
    - Iran
    - Mexico
    - Russia
    - South Africa $\leftarrow$ uses completely different labels

## Usage

To install dependencies and set up the python environment, run `uv sync`.

To setup all required intermediate data, run the numbered scripts in `scripts/` in numerical order. After this point,
any of the scripts can be run as seen fit (however, the test scripts obviously need a model to be trained first).

The following are the main scripts of interest:

- `models/*`: model definitions, not really intended to be run directly, but they do contain some test code
- `train/train_model.py`: train any model, including with/without fusion
- `tune/tune_model.py`: tune any model, again, including with/without fusion
- `test/test_model.py`: test a trained model on the test set, and make useful plots and tables
- `pred_fusion_from_image/*`: scripts related to predicting the environmental variables from RGB images
- `monitor.py`: a script to monitor the training progress of a model
- `dataset.py`: the main dataset class, which handles all data loading and preprocessing
- `make_report_plots.py`: make more polished plots (and latex tables) for the thesis
- `make_report_samples.py`: make the plots showing representative samples from each model/site

The approach taken to train/eval multiple models is to copy over `train/train_model.py` (or equivalent) into `tasks/`,
and rename it according to what exactly to do. Then, just modify the config parts of the scripts and run
`run_tasks.sh`.

All model classes inherit from the `Model` class (`scripts/models/model.py`), which also handles automatic model
loading. Therefore, for testing etc, just call `Model.load_model()` and pass the model path. All data related to a model
is stored within a specified folder, including plots specific to that model.

See `workflow.pdf` for a high-level overview of the workflow.