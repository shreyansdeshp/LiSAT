"""
Runs the trained model over every patch to build a probability heatmap, then joins it against the known-monuments layer to compute detected/missed sites. Saves a checkpoint of the heatmap + model so later stages don't need to retrain.

Original notebook cell(s): [21]
Depends on (run first / shares globals with): 06_train.py (model, CFG, LIDARMEAN, LIDARSTD, DEVICE)
"""

# ----- (notebook cell 21) -----
# ============================================================
# PREDICT + SPLIT MONUMENTS  (no plot — just prepares data for v9 and 3D)
# Run after training / benchmark (needs model, CFG, LIDARMEAN, LIDARSTD, DEVICE)
# Creates: heatmap, monuments_detected, monuments_missed, MON_THRESHOLD
# Also saves a checkpoint so you never have to rerun the model.
# ============================================================

from pathlib import Path
import json, shutil
import numpy as np
import torch
import rasterio
import geopandas as gpd
from shapely.geometry import box

MONUMENTS_PATH = Path("/kaggle/input/datasets/shreyansdeshpande/monuments/monuments_inside_valid_dtm.gpkg")
MON_THRESHOLD  = round(float(bestthresh), 2) if "bestthresh" in globals() else 0.58
BATCH          = 32

# ---- 1. every patch (train + val + test) ------------------------------------
patch_files = []
for folder in [CFG.TRAINSITE, CFG.TRAINNOSITE, CFG.VALSITE, CFG.VALNOSITE, CFG.TESTSITE, CFG.TESTNOSITE]:
    if folder.exists():
        patch_files += sorted(folder.glob(f"*{CFG.IMGSUFFIX}"))
print(f"Patches: {len(patch_files):,}")

# ---- 2. same preprocessing as training --------------------------------------
rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
rgb_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def prep(path):
    with rasterio.open(path) as src:
        img = src.read(masked=True).astype(np.float32).filled(0.0)
        b = src.bounds
    sat = np.clip(img[CFG.SATBANDS] / CFG.SAT_SCALE, 0.0, 1.0)
    lid = (img[CFG.LIDARBANDS] - LIDARMEAN[:, None, None]) / (LIDARSTD[:, None, None] + 1e-6)
    lid = np.nan_to_num(np.clip(lid, -CFG.LIDARCLIP, CFG.LIDARCLIP), nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(np.concatenate([sat, lid], 0).astype(np.float32))
    if CFG.IMAGENET_NORM:
        x[:3] = (x[:3] - rgb_mean) / rgb_std
    return x, box(b.left, b.bottom, b.right, b.top)

# ---- 3. predict (batched) ----------------------------------------------------
model.eval()
probs, geoms = [], []
with torch.no_grad():
    for s in range(0, len(patch_files), BATCH):
        xs, gs = zip(*[prep(p) for p in patch_files[s:s + BATCH]])
        xb = torch.stack(xs).to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=CFG.USEAMP):
            probs += torch.sigmoid(model(xb).squeeze(1)).float().cpu().tolist()
        geoms += gs
        if (s // BATCH) % 40 == 0:
            print(f"  {min(s + BATCH, len(patch_files)):,}/{len(patch_files):,}")

heatmap = gpd.GeoDataFrame({"filepath": [str(p) for p in patch_files], "probability": probs},
                           geometry=list(geoms), crs="EPSG:2157")
print(f"Predicted {len(heatmap):,} patches (mean p = {heatmap.probability.mean():.3f})")

# ---- 4. split monuments ------------------------------------------------------
if not MONUMENTS_PATH.exists():
    MONUMENTS_PATH = next(Path("/kaggle/input").rglob("monuments_inside_valid_dtm.gpkg"))
mons = gpd.read_file(MONUMENTS_PATH)
mons = mons.set_crs("EPSG:2157") if mons.crs is None else mons.to_crs("EPSG:2157")
mons = mons[mons.geometry.notna() & ~mons.geometry.is_empty].reset_index(drop=True)

tiles = heatmap.copy()
tiles["split"] = np.select([tiles.filepath.str.contains("/train/"), tiles.filepath.str.contains("/validation/"),
                            tiles.filepath.str.contains("/test/")], ["train", "val", "test"], "other")
pairs = gpd.sjoin(mons[["geometry"]], tiles, how="inner", predicate="intersects")
mons["max_prob"] = pairs.groupby(level=0)["probability"].max()
mons["max_prob_test"] = pairs[pairs.split == "test"].groupby(level=0)["probability"].max()

covered = mons["max_prob"].notna()
monuments_detected = mons[covered & (mons.max_prob >= MON_THRESHOLD)].copy()
monuments_missed   = mons[covered & (mons.max_prob <  MON_THRESHOLD)].copy()
t = mons["max_prob_test"].dropna()

print(f"\nMonuments: {len(mons):,}  |  threshold p ≥ {MON_THRESHOLD}")
print(f"  Detected: {len(monuments_detected):,}   Missed: {len(monuments_missed):,}   "
      f"Not on any patch: {int((~covered).sum()):,}")
print(f"  Recall (all patches):       {len(monuments_detected) / max(covered.sum(), 1):.1%}")
if len(t):
    print(f"  Recall (test patches only): {(t >= MON_THRESHOLD).mean():.1%}  ← honest number")

# ---- 5. checkpoint -----------------------------------------------------------
CK = Path("/kaggle/working/checkpoint"); CK.mkdir(parents=True, exist_ok=True)
if "ckptpath" in globals() and Path(ckptpath).exists():
    shutil.copy(ckptpath, CK / "convnext_tiny_multimodal_best.pth")
np.savez(CK / "lidar_stats.npz", mean=LIDARMEAN, std=LIDARSTD)
heatmap.to_file(CK / "heatmap_tiles.gpkg", driver="GPKG")
monuments_detected.to_file(CK / "monuments_detected.gpkg", driver="GPKG")
monuments_missed.to_file(CK / "monuments_missed.gpkg", driver="GPKG")
(CK / "settings.json").write_text(json.dumps({"MON_THRESHOLD": MON_THRESHOLD,
                                              "bestthresh": float(globals().get("bestthresh", 0.58))}))
print(f"\nCheckpoint saved to {CK} ✓  →  now run v9, then 3D")
