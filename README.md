# LiDAR Archaeological Site Detection (ConvNeXt multimodal)

Detects likely archaeological monument sites from multimodal raster patches
(satellite RGB+NIR bands + 7 LiDAR-derived terrain bands) using a
ConvNeXt-Tiny backbone adapted to 11 input channels, then turns the trained
model's predictions into a prediction heatmap, candidate new-site list, and
an optional 3D terrain viewer / web export.

The original work was developed as a single Kaggle notebook
(`notebooks/convnext_multimodal.ipynb`). For version control and review,
that notebook has been split cell-by-cell into a sequential pipeline of
scripts under [`src/`](src/) — see below for how the pieces fit together.

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

## Trained weights

The trained checkpoint (`convnext_tiny_multimodal_best.pth`, ~111MB) is not
committed here either. Training from scratch via `06_train.py` reproduces it
(`CFG` has the exact hyperparameters used).

## Setup

```bash
pip install -r requirements.txt
```

GPU with CUDA is recommended for training (`CFG.USEAMP` auto-enables mixed
precision when available) but not required.
