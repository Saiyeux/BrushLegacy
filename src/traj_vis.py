"""
traj_vis.py  —  Visualise robot trajectory NPZ files

Supports both the legacy Cobrush format (robot_path_MMDD_HHMMSS.npz) and the
newer StrokeForge format (cv_a_TIMESTAMP.npz).

Overview: all strokes as XY top-down view (canvas frame).
Stroke:   XY / XZ / YZ detail for a single stroke index.

Usage:
    # Overview PNG (auto-saved next to NPZ):
    python src/traj_vis.py --npz data/trajectories/robot_path_0609_120000.npz

    # Detail view for stroke 3:
    python src/traj_vis.py --npz data/trajectories/robot_path_0609_120000.npz --stroke 3

    # Interactive window:
    python src/traj_vis.py --npz data/trajectories/robot_path_0609_120000.npz --show
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── NPZ loading (supports both legacy and new formats) ───────────────────────

def _load_legacy(data) -> dict:
    """Load from Cobrush robot_path_*.npz format."""
    vertices = data["vertices"]          # list of 4×4 SE(3) matrices
    nln      = list(data["nln"])         # 1 = stroke endpoint, 0 = waypoint
    color    = data.get("color", None)
    return dict(vertices=vertices, nln=nln, color=color, format="legacy")


def _load_new(data) -> dict:
    """Load from StrokeForge cv_a_*.npz format."""
    vertices         = data["vertices"]
    nln              = list(data["nln"])
    hover_z          = float(data["hover_z"])   if "hover_z"   in data else None
    z_canvas         = float(data["z_canvas"])  if "z_canvas"  in data else None
    seg_frame_ranges = data["seg_frame_ranges"] if "seg_frame_ranges" in data else None
    refill_n         = int(data["refill_every"]) if "refill_every" in data else 5
    stroke_types     = data["stroke_types"]      if "stroke_types"   in data else None
    return dict(vertices=vertices, nln=nln,
                hover_z=hover_z, z_canvas=z_canvas,
                seg_frame_ranges=seg_frame_ranges,
                refill_n=refill_n, stroke_types=stroke_types,
                format="new")


def _load_cobrush_pro(data) -> dict:
    """Load from Cobrush Pro / BrushLegacy curves.npz format."""
    n = int(data["n_curves"])
    w = int(data["canvas_width"])
    h = int(data["canvas_height"])
    curves, colors, widths = [], [], []
    for i in range(n):
        curves.append(data[f"curve_{i}"].astype(float))
        colors.append(tuple(int(v) for v in data[f"color_{i}"]))
        widths.append(float(data[f"width_{i}"]) if f"width_{i}" in data else 6.0)
    return dict(curves=curves, colors=colors, widths=widths,
                canvas_width=w, canvas_height=h, format="cobrush_pro")


def _load_brushlegacy_v2(data) -> dict:
    """Load brushlegacy_v2 action-sequence NPZ (produced by traj_calc.py)."""
    n  = int(data["n_actions"])
    W  = int(data["canvas_width"])
    H  = int(data["canvas_height"])
    types   = data["action_types"]           # (N,) int32
    curves  = data["curves"]                 # (N,2,2) float32
    colors  = data["colors"]                 # (N,3) uint8
    widths  = data["widths"]                 # (N,) float32
    slots   = data["slots"]                  # (N,) int32
    pal_rgb = data.get("palette_colors", None)
    pal_names = data.get("palette_names", None)
    return dict(
        n=n, canvas_width=W, canvas_height=H,
        action_types=types, curves=curves,
        colors=colors, widths=widths, slots=slots,
        palette_colors=pal_rgb, palette_names=pal_names,
        format="brushlegacy_v2",
    )


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    fmt  = str(data.get("format", b"")) if "format" in data else ""
    if "brushlegacy_v2" in fmt or ("n_actions" in data and "action_types" in data):
        return _load_brushlegacy_v2(data)
    if "n_curves" in data:
        return _load_cobrush_pro(data)
    if "q_end1" in data or "end2tip" in data:
        return _load_legacy(data)
    return _load_new(data)


def split_segments(vertices, nln):
    """Split flat vertex list into per-stroke segments by nln markers."""
    segs, start = [], 0
    for i, marker in enumerate(nln):
        if marker == 1:
            segs.append(np.array(vertices[start:i + 1]))
            start = i + 1
    if start < len(vertices):
        segs.append(np.array(vertices[start:]))
    return segs


def _pos(seg):
    """Extract (N,3) positions from an (N,4,4) SE(3) trajectory segment."""
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[1:] == (4, 4):
        return seg[:, :3, 3]
    if seg.ndim == 2 and seg.shape[1] == 3:
        return seg
    return None


def _paint_mask(pos, hover_z):
    """True where the robot is in contact with the canvas."""
    if hover_z is None:
        return np.ones(len(pos), dtype=bool)
    return pos[:, 2] < (hover_z - 0.005)


# ── Canvas calibration (optional) ────────────────────────────────────────────

def _try_load_canvas_cal(npz_path: str):
    """Look for canvas.npy near the NPZ file or in standard locations."""
    search = [
        Path(npz_path).parent / "canvas.npy",
        Path(npz_path).parent.parent / "calibration" / "canvas.npy",
        Path(npz_path).parent.parent.parent / "data" / "calibration" / "canvas.npy",
    ]
    for p in search:
        if p.exists():
            try:
                cal = np.load(str(p), allow_pickle=True).item()
                return (np.array(cal["origin"]),
                        np.array(cal["xyz_rot"]),
                        float(cal["width_m"]),
                        float(cal["height_m"]))
            except Exception:
                pass
    return None


def _to_canvas_uv(pos, origin, xyz_rot):
    """Robot-frame (N,3) → canvas UV (N,2) in metres."""
    rel = pos - origin
    uv  = rel @ xyz_rot
    return uv[:, :2]


# ── Plots ─────────────────────────────────────────────────────────────────────

TYPE_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
TYPE_LABELS = {0: "long", 1: "split_long", 2: "short", 3: "tiny"}


def plot_brushlegacy_v2(npz_info: dict, out_path: str) -> None:
    """Visualise brushlegacy_v2 action-sequence NPZ.

    Three sub-plots:
      Left   — trajectory map: paint strokes (coloured) + transit (dashed grey)
               + dip (★) / wash (≈) event markers down the right edge.
      Centre — painted result: strokes rendered as rotated rectangles.
      Right  — action timeline: vertical bar coloured by palette slot,
               with wash/dip events marked.
    """
    import math
    import cv2  # noqa: F811

    ACTION_PAINT, ACTION_DIP, ACTION_WASH = 0, 1, 2

    types  = npz_info["action_types"]
    curves = npz_info["curves"]
    colors = npz_info["colors"]
    widths = npz_info["widths"]
    slots  = npz_info["slots"]
    W      = npz_info["canvas_width"]
    H      = npz_info["canvas_height"]
    n_all  = npz_info["n"]

    pal_rgb   = npz_info.get("palette_colors")
    pal_names = npz_info.get("palette_names")

    # Paint-action indices
    paint_idx = [i for i in range(n_all) if int(types[i]) == ACTION_PAINT]
    n_paint   = len(paint_idx)

    # ── Left: trajectory map ──────────────────────────────────────────────────
    traj = np.full((H, W, 3), 245, dtype=np.uint8)
    cv2.rectangle(traj, (0, 0), (W - 1, H - 1), (160, 160, 160), 1)

    prev_end = None
    for i in paint_idx:
        pts = curves[i].astype(np.float32)
        p0  = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        p1  = (int(round(pts[1, 0])), int(round(pts[1, 1])))
        if prev_end is not None:
            dx = p0[0] - prev_end[0];  dy = p0[1] - prev_end[1]
            dist = max(1, int(np.hypot(dx, dy)))
            for t in range(0, dist, 10):
                t0 = t / dist;  t1 = min((t + 6) / dist, 1.0)
                a  = (int(prev_end[0] + dx * t0), int(prev_end[1] + dy * t0))
                b  = (int(prev_end[0] + dx * t1), int(prev_end[1] + dy * t1))
                cv2.line(traj, a, b, (195, 195, 195), 1, cv2.LINE_AA)
        prev_end = p1

    for k, i in enumerate(paint_idx):
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        col_vis = (max(0, r - 50), max(0, g - 50), max(0, b - 50))
        pts = curves[i].astype(np.float32)
        p0  = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        p1  = (int(round(pts[1, 0])), int(round(pts[1, 1])))
        cv2.line(traj, p0, p1, col_vis, 2, cv2.LINE_AA)
        cv2.circle(traj, p0, 2, col_vis, -1, cv2.LINE_AA)

    # ── Centre: painted result ────────────────────────────────────────────────
    if [int(colors[i, 0]) for i in paint_idx]:
        mean_c = np.array([[colors[i] for i in paint_idx]]).mean(axis=1)[0].round().astype(np.uint8)
    else:
        mean_c = np.array([200, 200, 200], dtype=np.uint8)
    paint = np.tile(mean_c, (H, W, 1))

    for i in paint_idx:
        r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
        pts = curves[i].astype(np.float32)
        cx  = float(pts[:, 0].mean());  cy = float(pts[:, 1].mean())
        dx  = float(pts[1, 0] - pts[0, 0]);  dy = float(pts[1, 1] - pts[0, 1])
        length    = math.hypot(dx, dy) * 2
        angle_deg = math.degrees(math.atan2(dy, dx))
        bw        = max(4.0, float(widths[i]))
        rect      = ((cx, cy), (max(4.0, length), bw), angle_deg)
        box       = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(paint, [box], (r, g, b))
    cv2.rectangle(paint, (0, 0), (W - 1, H - 1), (180, 180, 180), 1)

    # ── Right: action timeline ────────────────────────────────────────────────
    # Vertical strip: each row = one action; coloured by palette slot
    TW, TH = 80, H   # timeline width, height
    timeline = np.full((TH, TW, 3), 230, dtype=np.uint8)

    # Colour lookup for palette slots
    if pal_rgb is not None:
        slot_colors = {i: (int(pal_rgb[i, 0]), int(pal_rgb[i, 1]), int(pal_rgb[i, 2]))
                       for i in range(len(pal_rgb))}
    else:
        from palette_cfg import PALETTE_RGB
        slot_colors = {i: PALETTE_RGB[i] for i in range(len(PALETTE_RGB))}

    row_h = max(1, TH // max(n_all, 1))
    current_slot_color = (200, 200, 200)

    for i in range(n_all):
        atype = int(types[i])
        slot  = int(slots[i])
        y0    = int(i * TH / n_all)
        y1    = int((i + 1) * TH / n_all)

        if atype == ACTION_DIP and slot >= 0:
            current_slot_color = slot_colors.get(slot, (200, 200, 200))
            # Dip marker: filled rectangle with border
            cv2.rectangle(timeline, (2, y0), (TW - 2, max(y0 + 1, y1 - 1)),
                          current_slot_color, -1)
            cv2.rectangle(timeline, (2, y0), (TW - 2, max(y0 + 1, y1 - 1)),
                          (80, 80, 80), 1)
            cv2.putText(timeline, "DIP", (4, max(y0 + 8, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1, cv2.LINE_AA)
        elif atype == ACTION_WASH:
            # Wash marker: blue stripe
            cv2.rectangle(timeline, (2, y0), (TW - 2, max(y0 + 1, y1 - 1)),
                          (180, 220, 255), -1)
            cv2.putText(timeline, "WASH", (4, max(y0 + 8, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 80, 160), 1, cv2.LINE_AA)
        else:  # PAINT
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            cv2.rectangle(timeline, (2, y0), (TW - 2, max(y0 + 1, y1 - 1)),
                          (r, g, b), -1)

    cv2.rectangle(timeline, (0, 0), (TW - 1, TH - 1), (120, 120, 120), 1)

    # ── Combine ───────────────────────────────────────────────────────────────
    # Arrays were built by writing (r,g,b) values directly via cv2 — they are
    # already in RGB order.  Matplotlib imshow interprets them as RGB, so no
    # conversion is needed.  (cv2.imwrite expects BGR, but we never write here.)
    div = np.full((H, 4, 3), 180, dtype=np.uint8)
    combined = np.hstack([traj, div, paint, div, timeline])

    fig, ax = plt.subplots(1, 1, figsize=(18, 7))
    ax.imshow(combined, origin="upper")
    ax.axis("off")

    n_dip  = sum(1 for t in types if int(t) == ACTION_DIP)
    n_wash = sum(1 for t in types if int(t) == ACTION_WASH)
    ax.set_title(
        f"{n_paint} paint  |  {n_dip} dip  |  {n_wash} wash  "
        f"({n_all} total actions)    "
        f"[left: trajectory map   centre: painted result   right: action timeline]",
        fontsize=10
    )

    # Palette legend
    if pal_rgb is not None and pal_names is not None:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=np.array(pal_rgb[i]) / 255,
                         label=str(pal_names[i]))
                   for i in range(len(pal_rgb))]
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  framealpha=0.85, ncol=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[vis] brushlegacy_v2 → {out_path}")
    plt.close()


def plot_cobrush_pro(npz_info: dict, out_path: str) -> None:
    """Visualise BrushLegacy curves NPZ — trajectory style (traj_vis aesthetic).

    Two sub-plots side by side:
      Left  — trajectory map: thin coloured lines + transit dashes + stroke numbers.
               White background, each stroke shown as a line from start→end.
      Right — painted result: strokes rendered at full brush width on white canvas,
               approximating what the finished painting looks like.
    """
    import cv2

    curves  = npz_info["curves"]
    colors  = npz_info["colors"]
    widths  = npz_info.get("widths", [6.0] * len(curves))
    W = npz_info["canvas_width"]
    H = npz_info["canvas_height"]
    n = len(curves)

    # ── Left: trajectory map ──────────────────────────────────────────────────
    traj = np.full((H, W, 3), 245, dtype=np.uint8)   # light-grey background

    # Canvas border
    cv2.rectangle(traj, (0, 0), (W - 1, H - 1), (160, 160, 160), 1)

    # Transit lines (dashed grey)
    prev_end = None
    for curve in curves:
        pts = np.asarray(curve, dtype=np.float32)
        p0 = (int(round(pts[0][0])), int(round(pts[0][1])))
        p1 = (int(round(pts[1][0])), int(round(pts[1][1])))
        if prev_end is not None:
            # Draw dashed line manually (every 6px on / 4px off)
            dx = p0[0] - prev_end[0];  dy = p0[1] - prev_end[1]
            dist = max(1, int(np.hypot(dx, dy)))
            for t in range(0, dist, 10):
                t0 = t / dist;  t1 = min((t + 6) / dist, 1.0)
                a = (int(prev_end[0] + dx * t0), int(prev_end[1] + dy * t0))
                b = (int(prev_end[0] + dx * t1), int(prev_end[1] + dy * t1))
                cv2.line(traj, a, b, (190, 190, 190), 1, cv2.LINE_AA)
        prev_end = p1

    # Paint strokes — thin line + start dot + index label
    for i, (curve, (r, g, b)) in enumerate(zip(curves, colors)):
        pts = np.asarray(curve, dtype=np.float32)
        p0 = (int(round(pts[0][0])), int(round(pts[0][1])))
        p1 = (int(round(pts[1][0])), int(round(pts[1][1])))
        col = (int(r), int(g), int(b))
        # Darken very light colours so they're visible on light background
        col_vis = tuple(max(0, c - 60) for c in col)
        cv2.line(traj, p0, p1, col_vis, 2, cv2.LINE_AA)
        cv2.circle(traj, p0, 3, col_vis, -1, cv2.LINE_AA)
        # Label every 5th stroke to avoid clutter
        if i % 5 == 0:
            cv2.putText(traj, str(i), (p0[0] + 3, p0[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, col_vis, 1, cv2.LINE_AA)

    # ── Right: painted result ─────────────────────────────────────────────────
    # Base coat: mean of all stroke colours (matches stroke_gen canvas init)
    if colors:
        mean_c = np.array(colors).mean(axis=0).round().astype(np.uint8)
    else:
        mean_c = np.array([220, 220, 220], dtype=np.uint8)
    paint = np.tile(mean_c, (H, W, 1))

    import math
    for curve, (r, g, b), bw in zip(curves, colors, widths):
        pts = np.asarray(curve, dtype=np.float32)
        cx  = float(pts[:, 0].mean())
        cy  = float(pts[:, 1].mean())
        dx  = float(pts[1, 0] - pts[0, 0])
        dy  = float(pts[1, 1] - pts[0, 1])
        length = math.hypot(dx, dy) * 2        # full stroke length (traj_calc stores half-length endpoints)
        angle_deg = math.degrees(math.atan2(dy, dx))
        width = max(4.0, float(bw))
        rect  = ((cx, cy), (max(4.0, length), width), angle_deg)
        box   = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(paint, [box], (int(r), int(g), int(b)))

    cv2.rectangle(paint, (0, 0), (W - 1, H - 1), (180, 180, 180), 1)

    # ── Combine into side-by-side figure ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(traj, origin="upper")
    axes[0].set_title(f"Trajectory map  ({n} strokes)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(paint, origin="upper")
    axes[1].set_title(f"Painted result  ({W}×{H} px)", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[vis] overview → {out_path}")
    plt.close()


def plot_overview(npz_info: dict, npz_path: str, out_path: str) -> None:
    segments  = split_segments(npz_info["vertices"], npz_info["nln"])
    hover_z   = npz_info.get("hover_z")
    refill_n  = npz_info.get("refill_n", 5)
    stroke_types = npz_info.get("stroke_types")

    cal = _try_load_canvas_cal(npz_path)
    figsize = (9, 9)
    if cal is not None:
        _, _, w_m, h_m = cal
        base = 9.0
        figsize = (base, base * h_m / w_m) if w_m >= h_m else (base * w_m / h_m, base)

    fig, ax = plt.subplots(figsize=figsize)
    default_colors = cm.tab20(np.linspace(0, 1, max(len(segments), 1)))

    for k, seg in enumerate(segments):
        pos = _pos(seg)
        if pos is None:
            continue
        pm = _paint_mask(pos, hover_z)

        if stroke_types is not None and k < len(stroke_types):
            col = TYPE_COLORS.get(int(stroke_types[k]), "#888888")
        else:
            col = default_colors[k % len(default_colors)]

        if cal is not None:
            origin, xyz_rot, _, _ = cal
            uv = _to_canvas_uv(pos, origin, xyz_rot)
            u, v = uv[:, 0], uv[:, 1]
        else:
            # Fallback: robot Y vs X (typical canvas orientation)
            u, v = pos[:, 1], pos[:, 0]

        if np.any(~pm):
            ax.plot(u[~pm], v[~pm], "--", color="#cccccc", lw=0.5)
        if np.any(pm):
            ax.plot(u[pm], v[pm], "-", color=col, lw=1.8)
            i0 = np.where(pm)[0][0]
            ax.text(u[i0], v[i0], str(k), fontsize=6, color=col,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec=col, lw=0.5, alpha=0.85))

    if cal is not None:
        _, _, w_m, h_m = cal
        ax.set_xlim(-0.01, w_m + 0.01)
        ax.set_ylim(-0.01, h_m + 0.01)
        ax.add_patch(plt.Rectangle((0, 0), w_m, h_m,
                     fill=False, edgecolor="#888888", lw=1.2, zorder=0))

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("canvas U (m)")
    ax.set_ylabel("canvas V (m)")
    ax.set_title(f"{len(segments)} strokes  (dip every {refill_n})")
    ax.grid(True, alpha=0.3)

    if stroke_types is not None:
        from matplotlib.lines import Line2D
        seen = {int(t) for t in stroke_types}
        handles = [Line2D([0], [0], color=TYPE_COLORS[t], lw=2, label=TYPE_LABELS[t])
                   for t in sorted(seen) if t in TYPE_COLORS]
        if handles:
            ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[vis] overview → {out_path}")
    plt.close()


def plot_stroke(npz_info: dict, stroke_idx: int, out_path: str) -> None:
    segments = split_segments(npz_info["vertices"], npz_info["nln"])
    hover_z  = npz_info.get("hover_z")

    if stroke_idx >= len(segments):
        print(f"[ERROR] stroke {stroke_idx} not in 0..{len(segments)-1}")
        return

    pos = _pos(segments[stroke_idx])
    if pos is None:
        print(f"[ERROR] could not extract positions from stroke {stroke_idx}")
        return

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    pm = _paint_mask(pos, hover_z)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xi, yi, xl, yl) in zip(axes, [
            (x, y, "X (m)", "Y (m)"),
            (x, z, "X (m)", "Z (m)"),
            (y, z, "Y (m)", "Z (m)")]):
        if np.any(~pm):
            ax.plot(xi[~pm], yi[~pm], "o--", color="#999999", ms=3, lw=0.8, label="hover")
        if np.any(pm):
            ax.plot(xi[pm], yi[pm], "o-", color="crimson", ms=4, lw=1.5, label="paint")
        for i in range(len(xi)):
            ax.annotate(str(i), (xi[i], yi[i]), fontsize=5, color="navy",
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Stroke {stroke_idx}  ({len(pos)} waypoints)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[vis] stroke {stroke_idx} → {out_path}")
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Visualise robot trajectory NPZ")
    p.add_argument("--npz",    required=True, help="Trajectory NPZ file")
    p.add_argument("--stroke", type=int, default=None,
                   help="Detail view for one stroke index (omit for overview)")
    p.add_argument("--show",   action="store_true",
                   help="Show interactive window (requires display)")
    args = p.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    npz_path = Path(args.npz)
    npz_info = load_npz(str(npz_path))
    format_  = npz_info.get("format", "?")
    stem     = npz_path.stem

    # ── BrushLegacy v2 action-sequence format ────────────────────────────────
    if format_ == "brushlegacy_v2":
        n_paint = sum(1 for t in npz_info["action_types"] if int(t) == 0)
        print(f"[NPZ] {npz_path.name}  format={format_}  "
              f"{npz_info['n']} actions  ({n_paint} paint)")
        out = npz_path.parent / f"{stem}_overview.png"
        import cv2  # noqa: F401 — needed by plot_brushlegacy_v2
        plot_brushlegacy_v2(npz_info, str(out))
        if args.show:
            plt.show()
        return

    # ── Cobrush Pro pixel-space format ───────────────────────────────────────
    if format_ == "cobrush_pro":
        print(f"[NPZ] {npz_path.name}  format={format_}  {len(npz_info['curves'])} curves")
        out = npz_path.parent / f"{stem}_overview.png"
        plot_cobrush_pro(npz_info, str(out))
        if args.show:
            plt.show()
        return

    # ── Legacy / StrokeForge SE(3) format ────────────────────────────────────
    segments  = split_segments(npz_info["vertices"], npz_info["nln"])
    hover_z   = npz_info.get("hover_z")
    print(f"[NPZ] {npz_path.name}  format={format_}  {len(segments)} strokes"
          + (f"  hover_z={hover_z:.4f}" if hover_z is not None else ""))

    if args.stroke is not None:
        out = npz_path.parent / f"{stem}_stroke{args.stroke:03d}.png"
        plot_stroke(npz_info, args.stroke, str(out))
    else:
        out = npz_path.parent / f"{stem}_overview.png"
        plot_overview(npz_info, str(npz_path), str(out))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
