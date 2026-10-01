"""
Packages the 2D heatmap and/or 3D terrain outputs into a web_export.zip for the frontend. Run after 12_prediction_heatmap.py and/or 13_dtm_3d_viewer.py.

Original notebook cell(s): [24]
Depends on (run first / shares globals with): 12_prediction_heatmap.py and/or 13_dtm_3d_viewer.py
"""

# ----- (notebook cell 24) -----
# ============================================================
# WEB EXPORT — 2D heatmap + 3D terrain (no 3D heatmap)
# Run AFTER v10 (for 2D) and/or the DTM cell (for 3D).
# Produces: /kaggle/working/web_export/web_export.zip
# ============================================================

import json, zipfile, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
from rasterio.warp import transform_bounds

WGS84 = "EPSG:4326"

# ---- dedicated output dir: no clash with v10's or DTM's OUTPUT_DIR ----
WEB_EXPORT_DIR = Path("/kaggle/working/web_export")
WEB_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

tmp = WEB_EXPORT_DIR / "_tmp"
if tmp.exists(): shutil.rmtree(tmp)
tmp.mkdir(parents=True, exist_ok=True)

# ---- sanity: which upstream cells' outputs are present? ----
have_v10 = all(n in globals() for n in
               ["tiles", "rgba", "minx", "miny", "maxx", "maxy"])
have_dtm = all(n in globals() for n in
               ["rgb", "valid", "bx0", "by0", "bx1", "by1", "res",
                "crs", "dem", "zmin", "zmax", "exag"])
print(f"v10 outputs in memory:  {'✅' if have_v10 else '❌'}")
print(f"DTM outputs in memory:  {'✅' if have_dtm else '❌'}")
assert have_v10 or have_dtm, "Nothing to export — run v10 and/or the DTM cell first."

# ============================================================
# PART A — 2D HEATMAP
# ============================================================
has_candidates = False
overlay_bounds_wgs = None
terrain_meta = None

if have_v10:
    tiles_crs = tiles.crs

    MAX_PNG_SIDE = 2048
    step_png = max(1, int(np.ceil(max(rgba.shape[0], rgba.shape[1]) / MAX_PNG_SIDE)))
    rgba_small = rgba[::step_png, ::step_png]
    Image.fromarray((np.clip(rgba_small, 0, 1) * 255).astype("uint8"), "RGBA") \
         .save(tmp / "heatmap_overlay.png", optimize=True)

    overlay_bounds_wgs = list(transform_bounds(tiles_crs, WGS84,
                                               minx, miny, maxx, maxy))

    patch_cols = [c for c in ["geometry", "probability", "p", "p_rule", "split", "label"]
                  if c in tiles.columns]
    patches = tiles[patch_cols].to_crs(WGS84).copy()
    patches = patches.rename(columns={"probability": "score", "p": "prob_calibrated"})
    patches.to_file(tmp / "patches.geojson", driver="GeoJSON")

    det = monuments_detected[["geometry"]].copy(); det["status"] = "detected"
    mis = monuments_missed[["geometry"]].copy();    mis["status"] = "missed"
    mon_all = pd.concat([det, mis], ignore_index=True)
    mon_all = gpd.GeoDataFrame(mon_all, geometry="geometry",
                               crs=monuments_detected.crs).to_crs(WGS84)
    mon_all["id"] = [f"M{i+1:04d}" for i in range(len(mon_all))]
    mon_all.to_file(tmp / "monuments.geojson", driver="GeoJSON")

    study_area.to_crs(WGS84).to_file(tmp / "study_area.geojson", driver="GeoJSON")

    has_candidates = "zones" in globals() and len(zones) > 0
    if has_candidates:
        zones.to_crs(WGS84).to_file(tmp / "candidate_sites.geojson", driver="GeoJSON")

    print(f"2D heatmap export ready  ({rgba_small.shape[1]}×{rgba_small.shape[0]} px)")

# ============================================================
# PART B — 3D TERRAIN
# ============================================================
if have_dtm:
    MAX_TERRAIN_SIDE = 2048
    step_t = max(1, int(np.ceil(max(rgb.shape[0], rgb.shape[1]) / MAX_TERRAIN_SIDE)))
    rgb_small = np.clip(rgb[::step_t, ::step_t], 0, 1)
    valid_small = valid[::step_t, ::step_t]
    rgba_terrain = np.dstack([rgb_small, valid_small.astype(float)])
    Image.fromarray((rgba_terrain * 255).astype("uint8"), "RGBA") \
         .save(tmp / "terrain_texture.png", optimize=True)

    MAX_GRID_SIDE = 1024
    step_g = max(1, int(np.ceil(max(dem.shape[0], dem.shape[1]) / MAX_GRID_SIDE)))
    dem_small = dem[::step_g, ::step_g]
    np.save(tmp / "terrain_grid.npy", dem_small.astype("float32"))

    terrain_meta = {
        "grid_file": "terrain_grid.npy",
        "texture_file": "terrain_texture.png",
        "shape": list(dem_small.shape),
        "resolution_m": float(res * step_g),
        "bounds_wgs84": list(transform_bounds(crs, WGS84, bx0, by0, bx1, by1)),
        "elevation_min_m": float(zmin),
        "elevation_max_m": float(zmax),
        "relief_m": float(zmax - zmin),
        "vertical_exaggeration": float(exag),
        "crs_projected": str(crs),
    }
    (tmp / "terrain_meta.json").write_text(json.dumps(terrain_meta, indent=2), encoding="utf-8")

    print(f"3D terrain export ready  ({dem_small.shape[1]}×{dem_small.shape[0]} grid, "
          f"relief {zmax-zmin:.0f} m, ×{exag:.1f} exaggeration)")

# ============================================================
# PART C — settings.json
# ============================================================
settings = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "views": [],
    "thresholds": {},
    "colormap": {},
    "confidence_classes": [],
    "counts": {},
}

if have_v10:
    settings["thresholds"] = {
        "rule_threshold": float(RULE_T),
        "mon_threshold":  float(MON_THRESHOLD),
        "bestthresh":     float(globals().get("bestthresh", MON_THRESHOLD)),
    }
    settings["colormap"] = {
        "name": "warm",
        "stops": [
            [0.00, "#ffe97a"], [0.20, "#ffd23f"], [0.38, "#fdb030"],
            [0.55, "#fb8a24"], [0.70, "#f05a1e"], [0.83, "#d8261b"],
            [0.93, "#a80f16"], [1.00, "#650010"],
        ],
        "label": "Predicted probability of a monument"
                 + (" (calibrated)" if CALIBRATE else ""),
    }
    settings["confidence_classes"] = [
        {"label": n_, "class_pct": float(r_), "score": float(sc_), "color_value": float(c_)}
        for sc_, r_, c_, n_ in CLASS_EDGES
    ] if NEIGHBOUR_RULE else []
    settings["counts"] = {
        "detected":   int(len(monuments_detected)),
        "missed":     int(len(monuments_missed)),
        "patches":    int(len(tiles)),
        "recall":     float(len(monuments_detected) /
                            max(len(monuments_detected) + len(monuments_missed), 1)),
        "candidates": int(len(zones)) if has_candidates else 0,
    }
    settings["views"].append({
        "id": "heatmap",
        "label": "Prediction map (2D)",
        "kind": "map2d",
        "crs_projected": str(tiles_crs),
        "crs_web": WGS84,
        "center_bounds_wgs84": overlay_bounds_wgs,
        "layers": [
            {"id": "svf",       "file": Path(globals().get(
                "STUDY_AREA_PATH",
                "/kaggle/input/datasets/shreyansdeshpande/svfblabla/finaldtm_SVF_R16_D16.tif"
            )).name, "type": "raster",
             "label": "Sky-View Factor basemap", "default_on": True},
            {"id": "heatmap",   "file": "heatmap_overlay.png", "type": "raster-overlay",
             "label": "Prediction heatmap", "default_on": True,
             "bounds_wgs84": overlay_bounds_wgs},
            {"id": "study",     "file": "study_area.geojson",  "type": "outline",
             "label": "Study area", "default_on": True, "color": "#222222"},
            {"id": "monuments", "file": "monuments.geojson",   "type": "points",
             "label": "Monuments", "default_on": True},
            {"id": "patches",   "file": "patches.geojson",     "type": "polygons",
             "label": "Model patches", "default_on": False},
            *([{"id": "candidates", "file": "candidate_sites.geojson", "type": "polygons",
                "label": "Candidate new sites", "default_on": True, "dashed": True}]
              if has_candidates else []),
        ],
    })

if have_dtm:
    settings["views"].append({
        "id": "terrain",
        "label": "DTM terrain (3D)",
        "kind": "terrain3d",
        "texture_file": "terrain_texture.png",
        "grid_file":    "terrain_grid.npy",
        "meta_file":    "terrain_meta.json",
        "bounds_wgs84": terrain_meta["bounds_wgs84"],
        "elevation_min_m": terrain_meta["elevation_min_m"],
        "elevation_max_m": terrain_meta["elevation_max_m"],
        "relief_m":        terrain_meta["relief_m"],
        "vertical_exaggeration": terrain_meta["vertical_exaggeration"],
        "resolution_m":    terrain_meta["resolution_m"],
    })

(tmp / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

# ============================================================
# PART D — zip
# ============================================================
zip_path = WEB_EXPORT_DIR / "web_export.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in sorted(tmp.iterdir()):
        z.write(f, f.name)
    # include SVF for the 2D basemap (with safe fallback)
    svf_src = Path(globals().get(
        "STUDY_AREA_PATH",
        "/kaggle/input/datasets/shreyansdeshpande/svfblabla/finaldtm_SVF_R16_D16.tif"
    ))
    if svf_src.exists():
        z.write(svf_src, svf_src.name)
    else:
        print(f"⚠ SVF not found at {svf_src} — zip will have no basemap")

shutil.rmtree(tmp, ignore_errors=True)

print(f"\n✅ web_export.zip  →  {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        print(f"   {info.filename:<28}  {info.file_size/1e6:6.2f} MB")
print("\n   Download: Kaggle → Output → web_export → web_export.zip")
print("   Unzip into  backend/data/")
