"""
Serializes the trained model, LiDAR normalization stats, heatmap/monument layers, thresholds, and raw patch caches to disk.

Original notebook cell(s): [25]
Depends on (run first / shares globals with): 06_train.py, 11_predict_monuments.py
"""

# ----- (notebook cell 25) -----
import shutil, json, pickle
from pathlib import Path
SAVE = Path("/kaggle/working/final_state")
SAVE.mkdir(parents=True, exist_ok=True)

# Model
shutil.copy(ckptpath, SAVE / "model.pth")

# LiDAR stats
np.savez(SAVE / "lidar_stats.npz", mean=LIDARMEAN, std=LIDARSTD)

# Heatmap + monuments (from PREDICT cell)
heatmap.to_file(SAVE / "heatmap_tiles.gpkg", driver="GPKG")
monuments_detected.to_file(SAVE / "monuments_detected.gpkg", driver="GPKG")
monuments_missed.to_file(SAVE / "monuments_missed.gpkg", driver="GPKG")

# Settings
json.dump({"bestthresh": float(bestthresh),
           "MON_THRESHOLD": float(MON_THRESHOLD)},
          open(SAVE / "settings.json", "w"))

# Optional: cache the raw patches too, so you never re-read .tif files
with open(SAVE / "cache_train.pkl", "wb") as f: pickle.dump(traincache, f)
with open(SAVE / "cache_val.pkl",   "wb") as f: pickle.dump(valcache,   f)
with open(SAVE / "cache_test.pkl",  "wb") as f: pickle.dump(testcache,  f)

print("Saved:")
for f in sorted(SAVE.iterdir()):
    print(" ", f.name, f.stat().st_size // 1024 // 1024, "MB")
