# LiDAR and Satellite Archaeological Site Detection (ConvNeXt multimodal)

## Overview

This project explores the use of **LiDAR-derived terrain data, Sentinel-2
satellite imagery, and a ConvNeXt-Tiny convolutional network** to identify
areas with archaeological potential in **Ireland**.

The goal is a deep-learning pipeline that analyses multiple geospatial
layers simultaneously — terrain geometry from LiDAR and spectral/surface
information from satellite imagery — and predicts whether a given patch of
ground is likely to contain an archaeological site, then turns those
per-patch predictions into a georeferenced **archaeological probability
heatmap** that can guide field surveys.

Traditional surveys require manually inspecting large geographic regions.
Automating the first pass over terrain and satellite characteristics lets
that manual effort focus on the areas the model flags as most promising.

The original work was developed as a single Kaggle notebook
(`notebooks/convnext_multimodal.ipynb`). For version control and review,
that notebook has been split cell-by-cell into a sequential pipeline of
scripts under [`src/`](src/) — see the "Pipeline overview" / "Repo layout"
sections below for how the pieces fit together.

## Data sources

### LiDAR / terrain data

A high-resolution Digital Terrain Model (DTM), at roughly **1 metre per
pixel**, captures subtle ground-elevation variation. Terrain visualisations
derived from it can reveal structures that are hard to spot in ordinary
satellite imagery — mounds, enclosures, earthworks, depressions, ancient
walls, and other terrain anomalies.

### Sentinel-2 satellite imagery

Sentinel-2 provides Red / Green / Blue / Near-Infrared bands at roughly
**10 metre resolution**, aligned to the LiDAR data before use.

## Input channels

Each patch fed to the model is an **11-channel, 200×200 pixel** GeoTIFF:

| Channels | Source | Count |
|---|---|---|
| Red, Green, Blue, Near-Infrared | Sentinel-2 | 4 |
| Slope, Sky-View Factor, Local Relief Model, Multi-directional Hillshade (4 directions) | DTM-derived | 7 |

All channels describe the exact same patch of ground, so the network sees
several complementary representations of each location at once — see
[`src/00_config.py`](src/00_config.py) (`CFG.SATBANDS` / `CFG.LIDARBANDS`)
for the exact band indices, and [`src/01_data.py`](src/01_data.py) for how
they're loaded and normalized.

## Labelling

The model performs **binary classification**. Each patch is labelled:

```text
0 -> no known archaeological site  (no_site/)
1 -> archaeological site present  (site/)
```

Known monument locations determine positive samples; areas without a
recorded monument nearby are used as negatives. Patches already arrive
pre-split into `train/`, `validation/`, and `test/` folders (see
[Dataset](#dataset) below) — this repo doesn't re-split them.

## Binary classification & thresholding

The model outputs a single raw **logit** per patch; a sigmoid converts it
to a probability. Rather than using a fixed 0.5 cutoff, the pipeline
calibrates a decision threshold on the validation set (maximizing F1 via
`precision_recall_curve`, in [`src/07_evaluate_test.py`](src/07_evaluate_test.py))
and reuses that threshold — `bestthresh` — for the test-set benchmark and
later for the full-area heatmap.

## Evaluation

Beyond standard accuracy / precision / recall / F1 / ROC-AUC / PR-AUC
([`src/05_engine.py`](src/05_engine.py), [`src/07_evaluate_test.py`](src/07_evaluate_test.py)),
this pipeline also checks whether the result is actually meaningful rather
than a fluke of a small, imbalanced test set
([`src/09_significance_tests.py`](src/09_significance_tests.py)):

- **Permutation test** — shuffles test labels thousands of times to build
  a null distribution and reports a p-value for "is the model better than
  chance?"
- **Bootstrap 95% confidence intervals** — resamples the test set to put
  error bars on ROC-AUC, PR-AUC, F1, precision, and recall.

Recall is treated as particularly important here: missing a real
archaeological site is costlier than flagging an occasional false
positive for a human to rule out.

## Pipeline overview

```
 INPUT                        TRAIN                         PREDICT
 ─────                        ─────                         ───────
 11-band patches      →   ConvNeXt-Tiny (11→3ch stem)  →   full-area inference
 (4 satellite +            + focal/BCE loss                 (every patch, not
  7 LiDAR bands,            + layer-wise LR decay             just train/val/test)
  200×200 .tif)             + early stopping on val AUC            │
      │                            │                               ▼
      │                            ▼                     per-patch probability
      │                   best checkpoint (.pth)         joined against known
      │                    + calibrated threshold         monument footprints
      │                                                            │
      ▼                                                            ▼
 train/val/test                                          PREDICTION HEATMAP
 split (site vs.                                          (probability surface
 no_site folders)                                          over the Sky-View-
                                                             Factor basemap,
                                                             confidence classes,
                                                             candidate new sites,
                                                             Kvamme's-gain accuracy)
                                                                    │
                                                                    ▼
                                                      optional: 3D DTM viewer,
                                                      web_export.zip for frontend
```

Areas with high predicted probability — and in particular the candidate
zones that have no known monument recorded nearby — are the ones worth
prioritizing for an actual field survey.

Stage → script mapping:

| Stage | Scripts |
|---|---|
| Input / data loading | `00_config.py`, `01_data.py` |
| ConvNeXt model + training | `02_model.py`, `03_losses.py`, `04_optim_utils.py`, `05_engine.py`, `06_train.py` |
| Evaluation on held-out test set | `07_evaluate_test.py`, `08_plots.py`, `09_significance_tests.py` |
| Full-area inference → heatmap | `11_predict_monuments.py`, `12_prediction_heatmap.py` |
| Optional extras | `10_train_one_run.py` (ablations), `13_dtm_3d_viewer.py`, `14_web_export.py`, `15_save_final_state.py` |

## Repo layout

```
notebooks/
  convnext_multimodal.ipynb   # original notebook, kept as the reference copy
src/
  00_config.py                 # CFG, device, seeding
  01_data.py                   # dataset indexing, raster caching, Dataset class
  02_model.py                  # ConvNeXt-Tiny -> 11-channel adaptation
  03_losses.py                 # optional focal loss
  04_optim_utils.py            # backbone freezing, layer-wise LR decay param groups
  05_engine.py                 # metrics, train/eval epoch loops
  06_train.py                  # MAIN entrypoint: caches, loaders, training loop
  07_evaluate_test.py          # final benchmark on the held-out test set
  08_plots.py                  # training curves, ROC/PR, F1-vs-threshold plots
  09_significance_tests.py     # permutation test + bootstrap 95% CIs
  10_train_one_run.py          # reusable train_one_run() for ablations/learning curves
  11_predict_monuments.py      # full-area inference -> heatmap + detected/missed monuments
  12_prediction_heatmap.py     # v10 prediction map figure, confidence classes, candidate sites
  13_dtm_3d_viewer.py          # standalone DTM 3D terrain viewer (Plotly + matplotlib)
  14_web_export.py             # packages 2D/3D outputs into web_export.zip
  15_save_final_state.py       # serializes model, stats, heatmap/monument layers to disk
```

### Why scripts instead of a package

The notebook cells share state through notebook globals (e.g. `model`,
`bestthresh`, `heatmap`, `tiles` get produced by one stage and consumed by
a later one) rather than explicit function arguments. Each `src/*.py` file
preserves that structure as-is — it's a straight split of the original
cells, not a rewrite — and documents its dependencies in its module
docstring. Run them **in order, in one shared namespace** (a Jupyter/IPython
session via `%run`, or `papermill`/`jupytext`), the same way the original
notebook cells were run top to bottom. `06_train.py` is the entrypoint for
training; `11`–`15` are post-training analysis/export stages and each notes
which earlier stage(s) it needs.

## Dataset

Patches are 200×200 GeoTIFFs with 11 bands (4 satellite + 7 LiDAR-derived),
organized as:

```
<dataset_root>/
  train/{site,no_site}/*.tif
  validation/{site,no_site}/*.tif
  test/{site,no_site}/*.tif
```

The dataset itself is **not** committed to this repo (several GB, well past
GitHub's file-size limits). It's hosted on Kaggle instead:

**[kaggle.com/datasets/shreyansdeshpande/200x200](https://www.kaggle.com/datasets/shreyansdeshpande/200x200)**

Download it from there and set `CFG.ROOT` in [`src/00_config.py`](src/00_config.py)
to wherever you've placed it locally (it defaults to the Kaggle-notebook
mount path, `/kaggle/input/datasets/shreyansdeshpande/200x200/FINALCNNDATA_200`,
which is used automatically if you run the pipeline inside a Kaggle notebook
with this dataset attached).

### Other input datasets (post-training / heatmap stages)

The monument-prediction, heatmap, and 3D-terrain stages (`11`–`14` in
`src/`) read a few more Kaggle-hosted inputs:

| Dataset | Used by | Kaggle-mount path in code |
|---|---|---|
| [`monuments`](https://www.kaggle.com/datasets/shreyansdeshpande/monuments) — known monument footprints (`monuments_inside_valid_dtm.gpkg`) | `11_predict_monuments.py`, `12_prediction_heatmap.py` | `/kaggle/input/datasets/shreyansdeshpande/monuments/monuments_inside_valid_dtm.gpkg` |
| [`dtm_data`](https://www.kaggle.com/datasets/shreyansdeshpande/dtmfile1234) — raw DTM tile(s) | `13_dtm_3d_viewer.py` | `/kaggle/input/datasets/shreyansdeshpande/dtmfile1234` |
| [`channel_stack`](https://www.kaggle.com/datasets/shreyansdeshpande/11stack) — the 11-band raster stack (Sky-View-Factor basemap source) | `12_prediction_heatmap.py`, `14_web_export.py` | NIL |

As with the patches dataset, these are read directly from Kaggle's
`/kaggle/input/` mount when run in a Kaggle notebook with the datasets
attached; outside Kaggle, download them and update the corresponding
path variables.

## Trained weights

The trained checkpoint (`convnext_tiny_multimodal_best.pth`, ~111MB) is not
committed here either. Training from scratch via `06_train.py` reproduces it
(`CFG` has the exact hyperparameters used).

## Technologies used

* Python
* PyTorch / torchvision — model, training loop
* NumPy / Pandas — array and tabular data handling
* Rasterio — reading/writing GeoTIFFs
* GeoPandas / Shapely — vector geometry (monuments, study area, candidate sites)
* SciPy — image filtering/smoothing for the heatmap and DTM viewer
* scikit-learn — metrics, calibration, bootstrap/permutation testing
* Matplotlib / Plotly — static and interactive visualization
* Pillow — PNG export for the web bundle
* Sentinel-2 imagery and LiDAR-derived DTMs — the underlying geospatial data

See [`requirements.txt`](requirements.txt) for exact packages.

## Setup

```bash
pip install -r requirements.txt
```

GPU with CUDA is recommended for training (`CFG.USEAMP` auto-enables mixed
precision when available) but not required.
