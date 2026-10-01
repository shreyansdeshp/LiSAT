"""
Builds the v10 prediction heatmap figure over the Sky-View-Factor basemap, including the 'neighbour rule' confidence classes, candidate new-site detection, and Kvamme's-gain accuracy table. Run after 11_predict_monuments.py.

Original notebook cell(s): [22]
Depends on (run first / shares globals with): 11_predict_monuments.py
"""

# ----- (notebook cell 22) -----
# ============================================================
# PREDICTION HEATMAP — v10
#   "Where are monuments likely to be?"  (the project's main result)
#   colour = model's CALIBRATED probability that a monument is there,
#            smoothed ~1 patch so neighbours share evidence
#   + known monuments on top (to check the map)
#   + candidate NEW sites (high probability, no recorded monument)
#   + accuracy on the TEST set (Kvamme's gain)
# Run AFTER "PREDICT + SPLIT MONUMENTS" (needs heatmap, monuments_detected,
# monuments_missed, MON_THRESHOLD). Run the 3D cell after this one to get
# the prediction map in 3D.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi
from shapely.geometry import box as shapely_box, shape as shapely_shape
from shapely.ops import unary_union
import rasterio
from rasterio.features import shapes as rasterio_shapes, rasterize, geometry_mask
from rasterio.enums import Resampling, MergeAlg
from rasterio.transform import from_origin
from sklearn.linear_model import LogisticRegression

# ============================================================
# PARAMETERS
# ============================================================
STUDY_AREA_PATH = "/kaggle/input/datasets/shreyansdeshpande/svfblabla/finaldtm_SVF_R16_D16.tif"
MONUMENTS_PATH  = "/kaggle/input/datasets/shreyansdeshpande/monuments/monuments_inside_valid_dtm.gpkg"

RESOLUTION_M = 20        # map grid
SMOOTH_M     = 150       # smoothing (≈ ¾ patch). 0 = raw patch squares, 250 = very soft
CALIBRATE    = True      # turn model scores into real probabilities (using validation patches)

P_LOW  = None            # below this probability: transparent (plain SVF).  None = average
                         #   chance of a patch holding a site (the "base rate")
P_HIGH = None            # at/above this: darkest red.  None = top 3 % of the area
HIGH_ZONE = None         # "high-probability zone" used for candidates + accuracy.
                         #   None = the model's validation-calibrated threshold

HAZE_ALPHA, CORE_ALPHA = 0.7, 0.58
SVF_LIGHTNESS = (0.18, 0.96)
SHOW_KNOWN    = True     # known monument outlines + detected/missed markers
SHOW_CANDIDATES = True   # dashed outlines around high-probability zones with no known monument
CANDIDATE_BUFFER_M = 100 # a zone counts as "new" if no known monument is within this distance
CANDIDATE_MIN_PATCHES = 2  # ignore isolated single high patches (usually noise)

# ---- NEIGHBOUR RULE (mentor's rule) -----------------------------------------
# Confidence classes run from the model's threshold (= "50 %") up to 1.0 (= "100 %"):
#     score = threshold + (class % − 50) / 50 × (1 − threshold)
#   e.g. threshold 0.585 → 50 % = 0.585 · 60 % = 0.668 · 70 % = 0.751 · 80 % = 0.834 · 90 % = 0.917
# Every patch in a class colours ITSELF and its 8 NEIGHBOURS in the class colour,
# with a soft fade around the outside.
NEIGHBOUR_RULE = True       # False = smooth probability map (previous v10 look)
RULE_USES      = "model"    # "model" = the model's own score · "calibrated" = calibrated probability
RULE_THRESHOLD = None       # None = the model's validation-calibrated threshold (bestthresh, e.g. 0.585)
SCALE_START    = 0.50       # the class % that equals the threshold (threshold = "50 %")
CLASSES = [                 # (class % from, colour value, name)
    (0.90, 1.00, "dark red"),
    (0.80, 0.80, "light red"),
    (0.70, 0.60, "orange"),
    (0.60, 0.24, "yellow"),
    (0.50, 0.08, "faint"),
]                           # below the last class (e.g. < 50 %) → not coloured
FADE_RING = True            # a faint halo one ring further out around red / orange classes
RULE_SHAPE    = "round"     # "round" = circular glow around each patch · "square" = 3×3 patch blocks
EDGE_SOFTEN_M = 30          # extra softening of the edges (0 = hard edges)

OUTPUT_DIR = Path("/kaggle/working/final_heatmap"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TITLE = "Archaeological Site Prediction Map"


# ============================================================
# 1. PATCH PROBABILITIES + CALIBRATION
# ============================================================
tiles = heatmap[["geometry", "probability", "filepath"]].copy()
fp = tiles["filepath"].str.replace("\\\\", "/", regex=True)
tiles["split"] = np.select([fp.str.contains("/train/"), fp.str.contains("/validation/"),
                            fp.str.contains("/test/")], ["train", "val", "test"], "other")
tiles["label"] = np.where(fp.str.contains("/no_site/"), 0, np.where(fp.str.contains("/site/"), 1, -1))

val = tiles[(tiles.split == "val") & (tiles.label >= 0)]
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
base_rate = float(val.label.mean()) if len(val) else float((tiles.label == 1).mean())

if CALIBRATE and len(val) > 50 and val.label.nunique() == 2:
    platt = LogisticRegression(C=1e6).fit(logit(val.probability.values)[:, None], val.label.values)
    calib = lambda p: platt.predict_proba(logit(np.asarray(p, float))[:, None])[:, 1]
    tiles["p"] = calib(tiles.probability.values)
    print(f"Calibrated on {len(val):,} validation patches "
          f"(average model score {val.probability.mean():.2f} → real site rate {base_rate:.2f})")
else:
    calib = lambda p: np.asarray(p, float)
    tiles["p"] = tiles.probability
    print("Calibration skipped — using raw model scores")

thr_raw = float(globals().get("bestthresh", MON_THRESHOLD))
HIGH = float(HIGH_ZONE) if HIGH_ZONE else float(calib([thr_raw])[0])
print(f"High-probability zone: calibrated p ≥ {HIGH:.2f}  (model score ≥ {thr_raw:.3f})")


# ============================================================
# 2. SVF BASEMAP + STUDY AREA  (same handling as v9)
# ============================================================
with rasterio.open(STUDY_AREA_PATH) as src:
    rb, rh, rw = src.bounds, src.height, src.width
    dec = max(1, int(np.ceil(max(rh, rw) / 2500)))
    oh, ow = max(1, rh // dec), max(1, rw // dec)
    svf = src.read(1, out_shape=(oh, ow), resampling=Resampling.bilinear,
                   masked=True).astype(np.float32).filled(np.nan)
    valid = src.dataset_mask(out_shape=(oh, ow)) > 0
    raw = src.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
    t = src.transform * src.transform.scale(rw / ow, rh / oh)
    raster_crs = src.crs or "EPSG:2157"
edge = np.r_[raw[0], raw[-1], raw[:, 0], raw[:, -1]]; edge = edge[np.isfinite(edge)]
if edge.size:
    vals, cnts = np.unique(edge, return_counts=True)
    if cnts.max() > 0.5 * edge.size:
        filled = np.isclose(raw, vals[cnts.argmax()]) & valid
        if filled.mean() > 0.01:
            valid &= ~filled; svf[filled] = np.nan
valid = ndi.binary_closing(ndi.binary_opening(valid, iterations=1), iterations=2)
px_ = abs(t.a)
polys = [shapely_shape(g) for g, v in rasterio_shapes(valid.astype(np.uint8), transform=t) if v == 1]
polys = [p for p in polys if p.area > 0.05e6]
merged_geom = (unary_union(polys).buffer(px_).buffer(-px_).simplify(px_ / 2) if polys
               else shapely_box(rb.left, rb.bottom, rb.right, rb.top))
study_area = gpd.GeoSeries([merged_geom], crs=raster_crs).to_crs(tiles.crs)
merged_geom = study_area.iloc[0].buffer(0)
vv = svf[np.isfinite(svf)]
svf_vmin, svf_vmax = np.percentile(vv, 2), np.percentile(vv, 98)
svf_cmap = LinearSegmentedColormap.from_list("svf", [str(SVF_LIGHTNESS[0]), str(SVF_LIGHTNESS[1])])


# ============================================================
# 3. PROBABILITY SURFACE  (patches → grid → smooth)
# ============================================================
pad = max(3 * SMOOTH_M, 400)
minx, miny, maxx, maxy = study_area.total_bounds
minx -= pad; miny -= pad; maxx += pad; maxy += pad
nx = int(np.ceil((maxx - minx) / RESOLUTION_M)); ny = int(np.ceil((maxy - miny) / RESOLUTION_M))
x_edges = minx + np.arange(nx + 1) * RESOLUTION_M
y_edges = miny + np.arange(ny + 1) * RESOLUTION_M
gt = from_origin(minx, maxy, RESOLUTION_M, RESOLUTION_M)

psum = rasterize(zip(tiles.geometry, tiles.p), out_shape=(ny, nx), transform=gt,
                 fill=0, merge_alg=MergeAlg.add, dtype="float32")
pcnt = rasterize(((g, 1.0) for g in tiles.geometry), out_shape=(ny, nx), transform=gt,
                 fill=0, merge_alg=MergeAlg.add, dtype="float32")
psum, pcnt = np.flipud(psum), np.flipud(pcnt)          # → south-up like the other maps
covered = pcnt > 0
if SMOOTH_M > 0:                                        # normalised smoothing: no edge darkening
    s = SMOOTH_M / RESOLUTION_M
    num = ndi.gaussian_filter(np.where(covered, psum / np.maximum(pcnt, 1), 0), s)
    den = ndi.gaussian_filter(covered.astype("float32"), s)
    prob = np.where(den > 0.25, num / np.maximum(den, 1e-6), np.nan)
else:
    prob = np.where(covered, psum / np.maximum(pcnt, 1), np.nan)

in_area = np.flipud(~geometry_mask([merged_geom], out_shape=(ny, nx), transform=gt))
prob[~in_area] = np.nan
ok = np.isfinite(prob)
lo = float(P_LOW) if P_LOW is not None else base_rate
hi = float(P_HIGH) if P_HIGH is not None else float(np.percentile(prob[ok], 97))
hi = max(hi, lo + 0.05)
value = np.clip((np.nan_to_num(prob, nan=0) - lo) / (hi - lo), 0, 1)   # 0–1 colour value

tiles["p_rule"] = tiles["probability"] if RULE_USES == "model" else tiles["p"]
RULE_T = float(RULE_THRESHOLD) if RULE_THRESHOLD is not None else float(
    globals().get("bestthresh", globals().get("MON_THRESHOLD", 0.5)))
if RULE_USES == "calibrated":
    RULE_T = float(calib([RULE_T])[0])
to_score = lambda r_: RULE_T + (r_ - SCALE_START) / (1 - SCALE_START) * (1 - RULE_T)   # class % → score
to_class = lambda sc: SCALE_START + (sc - RULE_T) / (1 - RULE_T) * (1 - SCALE_START)   # score → class %
TIERS = [(to_score(r_), [c_, c_] + ([0.06] if FADE_RING and c_ >= 0.6 else []))
         for r_, c_, _ in CLASSES]
CLASS_EDGES = sorted([(to_score(r_), r_, c_, n_) for r_, c_, n_ in CLASSES], reverse=True)
if NEIGHBOUR_RULE:
    # patch size in grid cells (patches are ~200 m → 10 cells at 20 m)
    psize = float(np.median(np.sqrt(tiles.geometry.area)))
    k = max(1, int(round(psize / RESOLUTION_M)))
    rule = np.zeros((ny, nx), dtype="float32")
    tiers_sorted = sorted(TIERS, key=lambda t_: -t_[0])
    upper = np.inf
    print(f"\nNeighbour rule on the {'model score' if RULE_USES == 'model' else 'calibrated probability'} "
          f"(patch ≈ {psize:.0f} m):")
    print(f"  threshold = {RULE_T:.3f}  →  {SCALE_START:.0%} = score {RULE_T:.3f}, 100 % = score 1.000")
    for lo_t, levels in tiers_sorted:
        sel = tiles[(tiles.p_rule >= lo_t) & (tiles.p_rule < upper)]
        rel = to_class(lo_t)
        name = next(n_ for sc_, r_, c_, n_ in CLASS_EDGES if abs(sc_ - lo_t) < 1e-9)
        print(f"  {rel:.0%}–{min(to_class(upper), 1):.0%} "
              f"(score {lo_t:.3f}–{min(upper, 1):.3f}) → {name:9s}: {len(sel):,} patches")
        if len(sel):
            if RULE_SHAPE == "round":
                # distance (m) from every map cell to the nearest confident patch CENTRE
                cpts = sel.geometry.centroid
                cc = np.clip(((cpts.x - minx) / RESOLUTION_M).astype(int), 0, nx - 1)
                cr = np.clip(((cpts.y - miny) / RESOLUTION_M).astype(int), 0, ny - 1)
                seeds = np.ones((ny, nx), bool); seeds[cr, cc] = False
                dist = ndi.distance_transform_edt(seeds) * RESOLUTION_M
                # ring radii chosen so each circle covers the same area as the square rings
                # (patch ≈ radius 0.56·P, 3×3 block ≈ 1.7·P, 5×5 ≈ 2.8·P, 7×7 ≈ 3.9·P)
                radii = psize * np.array([0.56, 1.13, 1.69, 2.25])[:len(levels)]
                # colour stays at each ring's level inside its circle, then blends to the next
                xp = np.r_[0.0, np.repeat(radii, 2)[:-1] + np.tile([0, 0.35 * psize], len(radii))[:-1],
                           radii[-1] + 0.5 * psize]
                fp = np.r_[levels[0], np.repeat(levels, 2)[1:], 0.0]
                order = np.argsort(xp, kind="stable")
                glow = np.interp(dist, xp[order], fp[order]).astype("float32")
                rule = np.maximum(rule, glow)
            else:
                centre = np.flipud(rasterize(((g, 1) for g in sel.geometry), out_shape=(ny, nx),
                                             transform=gt, fill=0, dtype="uint8")) > 0
                for ring, level in enumerate(levels):
                    m = centre if ring == 0 else ndi.maximum_filter(centre.astype("uint8"),
                                                                    size=2 * ring * k + 1) > 0
                    rule = np.maximum(rule, np.where(m, level, 0).astype("float32"))
        upper = lo_t
    if EDGE_SOFTEN_M > 0:
        rule = ndi.gaussian_filter(rule, EDGE_SOFTEN_M / RESOLUTION_M)
    value = np.where(in_area, rule, 0)
    ok = in_area.copy()
print(f"Colour scale: transparent below {lo:.0%} (average chance), darkest red at ≥ {hi:.0%}")


# ============================================================
# 4. ACCURACY ON THE TEST SET  (Kvamme's gain)
# ============================================================
known = gpd.read_file(MONUMENTS_PATH) if Path(MONUMENTS_PATH).exists() else \
        gpd.GeoDataFrame(geometry=list(monuments_detected.geometry) + list(monuments_missed.geometry),
                         crs=monuments_detected.crs)
known = known.set_crs(tiles.crs) if known.crs is None else known.to_crs(tiles.crs)
known = known[known.geometry.notna() & ~known.geometry.is_empty].reset_index(drop=True)

cent = gpd.GeoDataFrame(geometry=known.geometry.representative_point(), crs=tiles.crs)
test_tiles = tiles[tiles.split == "test"]
test_mon = gpd.sjoin(cent, test_tiles[["geometry"]], how="inner", predicate="within")
test_mon = test_mon[~test_mon.index.duplicated()]
ci = np.clip(((test_mon.geometry.x - minx) / RESOLUTION_M).astype(int), 0, nx - 1)
ri = np.clip(((test_mon.geometry.y - miny) / RESOLUTION_M).astype(int), 0, ny - 1)
field = value if NEIGHBOUR_RULE else prob                  # the map actually drawn
site_p = field[ri, ci]; site_p = site_p[np.isfinite(site_p)]

def gain_row(name, level):
    area = float((field[ok] >= level).mean())
    sites = float((site_p >= level).mean()) if site_p.size else np.nan
    g = 1 - area / sites if sites and sites > 0 else np.nan
    return {"zone": name, "level ≥": round(level, 3), "% of area": round(100 * area, 1),
            "% of test monuments": round(100 * sites, 1), "Kvamme gain": round(g, 2)}

if NEIGHBOUR_RULE:
    rows, names_ = [], []
    for sc_, r_, c_, n_ in CLASS_EDGES:                     # cumulative: dark red, + light red, …
        names_.append(n_)
        rows.append(gain_row(" + ".join(names_) if len(names_) < 3 else f"{names_[0]} … {n_}",
                             c_ * 0.85))
else:
    rows = [gain_row("high-probability zone", HIGH)]
    for pct in (10, 20, 30):                                  # top X % of the area
        rows.append(gain_row(f"top {pct}% of area", float(np.percentile(prob[ok], 100 - pct))))
gain_table = pd.DataFrame(rows)
print(f"\nAccuracy on {len(site_p)} TEST monuments (never seen in training):")
print(gain_table.to_string(index=False))
print("Gain: 1 = perfect, 0 = no better than random, > 0.5 = useful predictive model")
gain_table.to_csv(OUTPUT_DIR / "prediction_map_accuracy.csv", index=False)


# ============================================================
# 5. CANDIDATE NEW SITES  (high probability, no known monument nearby)
# ============================================================
CAND_LEVEL = CLASS_EDGES[min(1, len(CLASS_EDGES) - 1)][0] if NEIGHBOUR_RULE else HIGH   # ≥ 80 % relative
CAND_COL = "p_rule" if NEIGHBOUR_RULE else "p"
CAND_MIN = 1 if NEIGHBOUR_RULE else CANDIDATE_MIN_PATCHES   # one ≥ 80 % patch is already strong
hi_tiles = tiles[tiles[CAND_COL] >= CAND_LEVEL]
near_known = gpd.sjoin(hi_tiles, gpd.GeoDataFrame(geometry=known.buffer(CANDIDATE_BUFFER_M),
                                                  crs=tiles.crs), how="inner", predicate="intersects")
cand_tiles = hi_tiles[~hi_tiles.index.isin(near_known.index)]
if len(cand_tiles):
    zones = gpd.GeoDataFrame(geometry=[cand_tiles.buffer(1).union_all() if hasattr(cand_tiles, "union_all")
                                       else cand_tiles.buffer(1).unary_union], crs=tiles.crs)
    zones = zones.explode(index_parts=False).reset_index(drop=True)
    zones["geometry"] = zones.buffer(-1)
    j = gpd.sjoin(cand_tiles[["geometry", CAND_COL]], zones.reset_index().rename(columns={"index": "zone_id"}),
                  how="inner", predicate="intersects")
    stats = j.groupby("zone_id")[CAND_COL].agg(["max", "mean", "count"])
    zones["max_p"] = stats["max"].round(3); zones["mean_p"] = stats["mean"].round(3)
    zones["n_patches"] = stats["count"]; zones["area_km2"] = (zones.area / 1e6).round(3)
    zones["score"] = (zones["n_patches"] * zones["mean_p"]).round(2)       # size × strength
    zones = zones[zones["n_patches"] >= CAND_MIN]
    zones = zones.sort_values("score", ascending=False).reset_index(drop=True)
    zones["candidate"] = [f"C{i + 1}" for i in range(len(zones))]
    zones.to_file(OUTPUT_DIR / "candidate_new_sites.gpkg", driver="GPKG")
    zc = zones.copy(); zc["easting"] = zc.centroid.x.round(0); zc["northing"] = zc.centroid.y.round(0)
    zc.drop(columns="geometry").to_csv(OUTPUT_DIR / "candidate_new_sites.csv", index=False)
    print(f"\nCandidate new sites: {len(zones)} zones of ≥ {CAND_MIN} patches "
          f"({int(zones.n_patches.sum()) if len(zones) else 0} patches, "
          f"{zones.area_km2.sum():.1f} km²) with p ≥ {CAND_LEVEL:.2f} and no known monument within "
          f"{CANDIDATE_BUFFER_M} m → candidate_new_sites.gpkg / .csv")
else:
    zones = gpd.GeoDataFrame(geometry=[], crs=tiles.crs)
    print("\nNo candidate new sites at this threshold.")


# ============================================================
# 6. COLOURS  (warm only)
# ============================================================
glow_cmap = LinearSegmentedColormap.from_list("warm", [
    (0.00, "#ffe97a"), (0.20, "#ffd23f"), (0.38, "#fdb030"), (0.55, "#fb8a24"),
    (0.70, "#f05a1e"), (0.83, "#d8261b"), (0.93, "#a80f16"), (1.00, "#650010")])
MIN_SHOW = 0.02
rgba = glow_cmap(value)
fade = np.clip((value - MIN_SHOW) / (0.12 - MIN_SHOW), 0, 1)
rgba[..., 3] = np.where(ok, fade * (HAZE_ALPHA + (CORE_ALPHA - HAZE_ALPHA) * value), 0)


# ============================================================
# 7. FIGURE
# ============================================================
fig, ax = plt.subplots(figsize=(11, 12))
fig.patch.set_facecolor("white"); ax.set_facecolor("#dcdcdc")
svf_img = ax.imshow(svf, extent=[rb.left, rb.right, rb.bottom, rb.top], origin="upper",
                    cmap=svf_cmap, vmin=svf_vmin, vmax=svf_vmax, interpolation="bilinear", zorder=0)
heat_img = ax.imshow(rgba, extent=[minx, maxx, miny, maxy], origin="lower",
                     interpolation="bilinear", zorder=1)
study_area.boundary.plot(ax=ax, color="#222", linewidth=1.0, zorder=3)

def _poly_path(geom):
    polys_ = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    verts, codes = [], []
    for p in polys_:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords); verts.extend(xy)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)
clip_patch = PathPatch(_poly_path(merged_geom), transform=ax.transData, facecolor="none", edgecolor="none")
ax.add_patch(clip_patch); heat_img.set_clip_path(clip_patch); svf_img.set_clip_path(clip_patch)

if SHOW_CANDIDATES and len(zones):
    zones.boundary.plot(ax=ax, color="#1a1a1a", linewidth=1.1, linestyle=(0, (3, 2)), zorder=4)
    for _, zr in zones.head(10).iterrows():                     # label the 10 strongest
        c = zr.geometry.representative_point()
        ax.text(c.x, c.y, zr.candidate, fontsize=7, fontweight="bold", ha="center", va="center",
                zorder=9, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

if SHOW_KNOWN:
    kt = known.geom_type
    known[kt.isin(["Polygon", "MultiPolygon"])].boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=5)
    if kt.isin(["LineString", "MultiLineString"]).any():
        known[kt.isin(["LineString", "MultiLineString"])].plot(ax=ax, color="black", linewidth=1.1, zorder=5)
    dp = monuments_detected.geometry.centroid; mp = monuments_missed.geometry.centroid
    ax.scatter(dp.x, dp.y, s=6, c="#111", linewidths=0, alpha=0.8, zorder=6)
    ax.scatter(mp.x, mp.y, s=34, facecolors="none", edgecolors="#111", linewidths=3.2, zorder=7)
    ax.scatter(mp.x, mp.y, s=34, facecolors="none", edgecolors="white", linewidths=2.0, zorder=7)
    ax.scatter(mp.x, mp.y, s=34, facecolors="none", edgecolors="#e0001b", linewidths=1.0, zorder=8)

# colour bar in probability units, fading to grey at the low end like the map
t_ = np.linspace(0, 1, 256)
a_ = np.clip((t_ - MIN_SHOW) / (0.12 - MIN_SHOW), 0, 1) * (HAZE_ALPHA + (CORE_ALPHA - HAZE_ALPHA) * t_)
bar = glow_cmap(t_)[:, :3] * a_[:, None] + np.array([0.85, 0.85, 0.85]) * (1 - a_[:, None])
if not NEIGHBOUR_RULE:
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=ListedColormap(bar)),
                        ax=ax, fraction=0.035, pad=0.02)
    ticks = np.linspace(0, 1, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"≤ {lo:.0%}"] + [f"{lo + v * (hi - lo):.0%}" for v in ticks[1:-1]] + [f"≥ {hi:.0%}"])
    cbar.set_label("Predicted probability of a monument" + (" (calibrated)" if CALIBRATE else ""), fontsize=9)
    cbar.ax.tick_params(labelsize=8)
else:
    # colour key for the neighbour rule (a separate legend box on the right)
    def _shown(level):                      # colour as it appears on the map (with its transparency)
        a = np.clip((level - MIN_SHOW) / (0.12 - MIN_SHOW), 0, 1) * (HAZE_ALPHA + (CORE_ALPHA - HAZE_ALPHA) * level)
        return tuple(np.array(glow_cmap(level)[:3]) * a + np.array([0.85, 0.85, 0.85]) * (1 - a))
    who = "model score" if RULE_USES == "model" else "calibrated probability"
    key = []
    up = None
    for sc_, r_, c_, n_ in CLASS_EDGES:
        rtxt = f"{r_:.0%}–{(up if up is not None else 1):.0%}"
        stxt = f"score {sc_:.3f}–{to_score(up) if up is not None else 1:.3f}"
        key.append(Patch(facecolor=_shown(c_), edgecolor="#555", linewidth=0.4,
                         label=f"{rtxt}  {n_}  ({stxt})"))
        up = r_
    if FADE_RING:
        key.append(Patch(facecolor=_shown(0.06), edgecolor="#555", linewidth=0.4,
                         label="faint halo around red / orange zones"))
    who = f"{SCALE_START:.0%} = threshold {RULE_T:.3f}; each patch + its 8 neighbours"
    fig.subplots_adjust(right=0.74)
    fig.legend(handles=key, loc="upper left", bbox_to_anchor=(0.755, 0.88), fontsize=7.5,
               title=f"Confidence class\n({who})", title_fontsize=7.5, frameon=True, framealpha=0.95)

hz = gain_table.iloc[-1] if NEIGHBOUR_RULE else gain_table.iloc[0]   # rule: all coloured classes
handles = [Line2D([0], [0], color="#222", lw=1.2, label="Study area")]
if SHOW_CANDIDATES and len(zones):
    handles.append(Line2D([0], [0], color="#1a1a1a", lw=1.1, linestyle=(0, (3, 2)),
                          label=f"Candidate new site (score ≥ {CAND_LEVEL:.3f}, no known monument) — {len(zones)}"))
if SHOW_KNOWN:
    handles += [Line2D([0], [0], color="black", lw=0.8, label="Known monument"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#111", markeredgecolor="none",
                       markersize=5, label=f"Detected — {len(monuments_detected)}"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#d7191c",
                       markeredgewidth=1.3, markersize=7, label=f"Missed — {len(monuments_missed)}")]
ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.95,
          title=(f"Test set: {hz['% of test monuments']:.0f}% of monuments in "
                 f"{hz['% of area']:.0f}% of area (gain {hz['Kvamme gain']:.2f})"), title_fontsize=8)

sxmin, symin, sxmax, symax = study_area.total_bounds
pxm, pym = 0.03 * (sxmax - sxmin), 0.03 * (symax - symin)
ax.set_xlim(sxmin - pxm, sxmax + pxm); ax.set_ylim(symin - pym, symax + pym); ax.set_aspect("equal")
x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); span = x1 - x0
barlen = [1000, 2000, 5000, 10000][int(np.argmin(np.abs(np.array([1000, 2000, 5000, 10000]) - span / 5)))]
bx, by, bh = x0 + 0.04 * span, y0 + 0.04 * (y1 - y0), 0.006 * (y1 - y0)
ax.add_patch(Rectangle((bx - 0.02 * span, by - 1.5 * bh), barlen + 0.04 * span, 6 * bh,
                       facecolor="white", edgecolor="none", alpha=0.8, zorder=6.5))
ax.add_patch(Rectangle((bx, by), barlen / 2, bh, facecolor="#222", edgecolor="#222", zorder=7))
ax.add_patch(Rectangle((bx + barlen / 2, by), barlen / 2, bh, facecolor="white", edgecolor="#222", zorder=7))
ax.text(bx, by + 2.2 * bh, "0", ha="center", fontsize=8, zorder=7)
ax.text(bx + barlen, by + 2.2 * bh, f"{barlen / 1000:g} km", ha="center", fontsize=8, zorder=7)
ax.annotate("N", xy=(x0 + 0.06 * span, y1 - 0.05 * (y1 - y0)), xytext=(x0 + 0.06 * span, y1 - 0.12 * (y1 - y0)),
            ha="center", va="center", fontsize=12, fontweight="bold", zorder=7,
            arrowprops=dict(facecolor="#222", edgecolor="#222", width=4, headwidth=12))
ax.set_title(f"{TITLE}\n" + (f"Confidence above the {RULE_T:.3f} threshold · each patch + 8 neighbours"
                              if NEIGHBOUR_RULE else
                              f"ConvNeXt-Tiny prediction over Sky-View Factor (smoothed {SMOOTH_M} m)"),
             fontsize=12, fontweight="bold")
ax.set_xlabel("Easting (EPSG:2157)", fontsize=9); ax.set_ylabel("Northing (EPSG:2157)", fontsize=9)
ax.tick_params(labelsize=8); ax.ticklabel_format(style="plain", useOffset=False)
if not NEIGHBOUR_RULE:
    plt.tight_layout()
png_path = OUTPUT_DIR / "prediction_map_v10.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {png_path}")


# ============================================================
# 8. HAND OVER TO THE 3D CELL  (3D will now show this prediction surface)
# ============================================================
if NEIGHBOUR_RULE:
    density_km2 = value.astype("float32")                 # the rule's 0–1 colour value
    MAP_CBAR_LABEL = f"Confidence above threshold {RULE_T:.3f}"
    MAP_TICKTEXT = [f"{SCALE_START:.0%} (score {RULE_T:.2f})", "70 %", "90–100 %"]
else:
    density_km2 = np.clip((np.nan_to_num(prob, nan=0) - lo) / (hi - lo), 0, None).astype("float32")
    # (not capped at 1 → the strongest areas still rise above the rest in 3D)
    MAP_CBAR_LABEL = "Predicted probability of a monument"
    MAP_TICKTEXT = [f"≤ {lo:.0%}", f"{(lo + hi) / 2:.0%}", f"≥ {hi:.0%}"]
vmax, GAMMA = 1.0, 1.0
MAP_SUBTITLE = "Height & colour = predicted probability of a monument  ·  ground = DTM with Sky-View Factor shading"
