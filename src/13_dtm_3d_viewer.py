"""
Standalone DTM 3D viewer: mosaics DTM tiles and renders an interactive Plotly terrain, a matplotlib 3D render, and a 2D shaded-relief map.

Original notebook cell(s): [23]
Depends on (run first / shares globals with): Standalone (only needs the DTM raster path)
"""

# ----- (notebook cell 23) -----
# ============================================================
# DTM 3D VIEWER — stand-alone (needs nothing from other cells)
#   Reads your DTM (one file or a folder of tiles), then shows:
#     1. an interactive 3D terrain you can rotate / zoom (Plotly)
#     2. a high-resolution 3D image (matplotlib, hillshaded)
#     3. a 2D shaded-relief map for reference
# ============================================================

from pathlib import Path
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
import plotly.graph_objects as go

# ============================================================
# PARAMETERS
# ============================================================
DTM_PATH   = Path("/kaggle/input/datasets/shreyansdeshpande/dtm_data")   # file OR folder of tiles
MAX_PIXELS = 700          # longest side of the 3D grid (500–900). Bigger = more detail, slower
VERT_EXAG  = "auto"       # "auto" = hills clearly visible · 1 = true scale · e.g. 3, 5, 10
AUTO_RELIEF_SHARE = 0.12  # "auto": tallest relief drawn as ~12 % of the map width
COLORS     = "terrain"    # "terrain" (green → brown → white) or "earth" (muted, realistic)
VIEW_ELEV, VIEW_AZIM = 35, -60     # static image camera
OUTPUT_DIR = Path("/kaggle/working/dtm_3d"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. FIND FILES + COMMON GRID
# ============================================================
if DTM_PATH.is_file():
    files = [DTM_PATH]
else:
    files = sorted({p for e in ("*.tif", "*.tiff", "*.TIF", "*.vrt", "*.img") for p in DTM_PATH.rglob(e)})
if not files:
    raise FileNotFoundError(f"No DTM rasters found in {DTM_PATH}")
print(f"DTM files: {len(files)}")

with rasterio.open(files[0]) as s0:
    crs = s0.crs
bounds = []
for f in files:
    with rasterio.open(f) as s:
        bounds.append(transform_bounds(s.crs, crs, *s.bounds) if s.crs != crs else tuple(s.bounds))
bx0 = min(b[0] for b in bounds); by0 = min(b[1] for b in bounds)
bx1 = max(b[2] for b in bounds); by1 = max(b[3] for b in bounds)
res = max(bx1 - bx0, by1 - by0) / MAX_PIXELS
nx, ny = int(np.ceil((bx1 - bx0) / res)), int(np.ceil((by1 - by0) / res))
grid_t = from_origin(bx0, by1, res, res)
print(f"Area: {(bx1 - bx0) / 1000:.1f} × {(by1 - by0) / 1000:.1f} km  →  grid {ny} × {nx} at {res:.0f} m")


# ============================================================
# 2. READ + MOSAIC  (handles 'no data' and constant fill values)
# ============================================================
acc = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
for f in files:
    with rasterio.open(f) as src:
        nod = src.nodata
        if nod is None:          # no nodata set → look for a constant fill along the raster edge
            sm = src.read(1, out_shape=(max(1, src.height // 20), max(1, src.width // 20)),
                          resampling=Resampling.nearest)
            e = np.r_[sm[0], sm[-1], sm[:, 0], sm[:, -1]]
            v, c = np.unique(e, return_counts=True)
            nod = float(v[c.argmax()]) if c.max() > 0.5 * e.size else -9999
        with WarpedVRT(src, crs=crs, transform=grid_t, width=nx, height=ny,
                       resampling=Resampling.average, src_nodata=nod,
                       nodata=np.nan, dtype="float32") as vrt:
            a = vrt.read(1).astype("float64")
    a[(a < -500) | (a > 9000)] = np.nan
    ok = np.isfinite(a); acc[ok] += a[ok]; cnt[ok] += 1
dem = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)          # north-up
valid = np.isfinite(dem)
# fill tiny holes so the surface is continuous (big empty areas stay empty)
holes, n_h = ndi.label(~valid)
if n_h:
    sizes = ndi.sum(np.ones_like(holes), holes, range(1, n_h + 1))
    edge_lbl = set(np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
    small = np.isin(holes, [i + 1 for i, s in enumerate(sizes) if s < 50 and (i + 1) not in edge_lbl])
    small |= ndi.binary_closing(valid, iterations=3) & ~valid        # thin seams between tiles
    if small.any():
        w = ndi.gaussian_filter(valid.astype(float), 2); vv = ndi.gaussian_filter(np.nan_to_num(dem), 2)
        dem[small] = (vv / np.maximum(w, 1e-6))[small]; valid |= small

zmin, zmax = np.nanmin(dem), np.nanmax(dem)
relief = zmax - zmin
print(f"Elevation: {zmin:.1f} – {zmax:.1f} m  (relief {relief:.1f} m)")


# ============================================================
# 3. VERTICAL EXAGGERATION
# ============================================================
width_m = max(nx, ny) * res
exag = (AUTO_RELIEF_SHARE * width_m / max(relief, 1)) if VERT_EXAG == "auto" else float(VERT_EXAG)
exag = float(np.clip(exag, 1, 60))
print(f"Vertical exaggeration: ×{exag:.1f}  "
      f"({'auto' if VERT_EXAG == 'auto' else 'fixed'}; ×1 would look almost flat for {relief:.0f} m of relief "
      f"over {width_m / 1000:.0f} km)")


# ============================================================
# 4. COLOURS  (elevation tint × multi-direction hillshade)
# ============================================================
if COLORS == "earth":
    cmap = LinearSegmentedColormap.from_list("earth", [
        "#4f6b3a", "#6f8a4a", "#98a060", "#b9ae7a", "#c9b48f", "#d9cbb0", "#efe9df"])
else:
    cmap = LinearSegmentedColormap.from_list("terrain_nice", [
        (0.00, "#2e6b3a"), (0.15, "#4f8a3d"), (0.35, "#9bb05a"), (0.55, "#d8c47c"),
        (0.75, "#a8784a"), (0.90, "#8a6a55"), (1.00, "#f4f1ea")])

dem_f = np.where(valid, dem, np.nanmedian(dem))
ls_list = [(315, 0.45), (270, 0.2), (0, 0.2), (225, 0.15)]
hs = sum(wt * LightSource(azdeg=az, altdeg=40).hillshade(dem_f, vert_exag=exag, dx=res, dy=res)
         for az, wt in ls_list)
elev_n = np.clip((dem_f - zmin) / max(relief, 1e-6), 0, 1)
rgb = cmap(elev_n)[..., :3] * (0.35 + 0.8 * hs[..., None])
rgb = np.clip(rgb, 0, 1)

xs = bx0 + (np.arange(nx) + 0.5) * res
ys = by1 - (np.arange(ny) + 0.5) * res                       # north-up rows
Z = np.where(valid, (dem - zmin) * exag, np.nan)


# ============================================================
# 5. 2D SHADED RELIEF (reference)
# ============================================================
fig2, ax2 = plt.subplots(figsize=(9, 9))
ax2.imshow(np.dstack([rgb, valid.astype(float)]), extent=[bx0, bx0 + nx * res, by1 - ny * res, by1])
ax2.set_title(f"DTM shaded relief — {zmin:.0f} to {zmax:.0f} m", fontweight="bold")
ax2.set_xlabel("Easting"); ax2.set_ylabel("Northing"); ax2.ticklabel_format(style="plain", useOffset=False)
sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(zmin, zmax))
fig2.colorbar(sm_, ax=ax2, fraction=0.035, pad=0.02, label="Elevation (m)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "dtm_shaded_relief.png", dpi=250, bbox_inches="tight"); plt.show()


# ============================================================
# 6. INTERACTIVE 3D  (Plotly — rotate / zoom / hover shows real elevation)
# ============================================================
step = max(1, int(np.ceil(max(nx, ny) / 450)))              # keep the browser responsive
Zs = Z[::step, ::step]; Es = np.where(valid, dem, np.nan)[::step, ::step]
fig3 = go.Figure(go.Surface(
    x=xs[::step], y=ys[::step], z=Zs, surfacecolor=Es,
    colorscale=[[t, "#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(t)[:3])] for t in np.linspace(0, 1, 11)],
    cmin=zmin, cmax=zmax, colorbar=dict(title=dict(text="Elevation (m)", side="right"), len=0.6),
    lighting=dict(ambient=0.45, diffuse=0.85, specular=0.08, roughness=0.9, fresnel=0.1),
    lightposition=dict(x=-5e4, y=5e4, z=4e4),
    customdata=Es, hovertemplate="E %{x:.0f}<br>N %{y:.0f}<br>elevation %{customdata:.1f} m<extra></extra>"))
span_x, span_y = nx * res, ny * res
fig3.update_layout(
    title=f"DTM in 3D (vertical exaggeration ×{exag:.1f})", width=1150, height=820,
    margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="#e9eef2",
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
               bgcolor="#e9eef2", aspectmode="manual",
               aspectratio=dict(x=span_x / max(span_x, span_y), y=span_y / max(span_x, span_y),
                                z=float(np.nanmax(Zs)) / max(span_x, span_y)),
               camera=dict(eye=dict(x=0.9, y=-1.4, z=0.9))))
fig3.write_html(OUTPUT_DIR / "dtm_3d_interactive.html", include_plotlyjs=True)
fig3.show()


# ============================================================
# 7. HIGH-RESOLUTION 3D IMAGE  (matplotlib)
# ============================================================
st2 = max(1, int(np.ceil(max(nx, ny) / 600)))
Xg, Yg = np.meshgrid(xs[::st2], ys[::st2])
Zg = Z[::st2, ::st2]
fc = np.dstack([rgb[::st2, ::st2], valid[::st2, ::st2].astype(float)])
fig = plt.figure(figsize=(14, 10), facecolor="#eef2f5")
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
ax.set_facecolor("#eef2f5")
ax.plot_surface(Xg, Yg, np.where(np.isfinite(Zg), Zg, np.nan), facecolors=fc,
                rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
ax.set_box_aspect((span_x, span_y, float(np.nanmax(Zg))), zoom=1.25)
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
ax.set_axis_off()
ax.set_title(f"DTM in 3D — {zmin:.0f} to {zmax:.0f} m, vertical exaggeration ×{exag:.1f}",
             fontsize=14, fontweight="bold", pad=0)
plt.savefig(OUTPUT_DIR / "dtm_3d.png", dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"Saved to {OUTPUT_DIR}: dtm_shaded_relief.png, dtm_3d.png, dtm_3d_interactive.html")
