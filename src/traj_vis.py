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


def load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
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


def plot_cobrush_pro(npz_info: dict, out_path: str) -> None:
    """Visualise BrushLegacy curves NPZ as trajectory-style thick coloured lines.

    Paint strokes are drawn as thick coloured lines (brush colour).
    Transit moves between strokes are shown as thin dashed grey lines.
    This matches the traj_vis aesthetic: show where the robot goes, not a flat mosaic.
    """
    import cv2

    curves  = npz_info["curves"]
    colors  = npz_info["colors"]
    widths  = npz_info.get("widths", [6.0] * len(curves))
    W = npz_info["canvas_width"]
    H = npz_info["canvas_height"]

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)   # white background

    # Draw transit lines first (beneath paint strokes)
    prev_end = None
    for curve in curves:
        pts = np.asarray(curve, dtype=np.float32)
        p0 = (int(round(pts[0][0])), int(round(pts[0][1])))
        p1 = (int(round(pts[1][0])), int(round(pts[1][1])))
        if prev_end is not None:
            cv2.line(canvas, prev_end, p0, (210, 210, 210), 1, cv2.LINE_AA)
        prev_end = p1

    # Draw paint strokes as thick coloured lines
    for curve, (r, g, b), bw in zip(curves, colors, widths):
        pts = np.asarray(curve, dtype=np.float32)
        p0 = (int(round(pts[0][0])), int(round(pts[0][1])))
        p1 = (int(round(pts[1][0])), int(round(pts[1][1])))
        thickness = max(2, int(round(bw)))
        cv2.line(canvas, p0, p1, (r, g, b), thickness, cv2.LINE_AA)

    # Canvas border
    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (180, 180, 180), 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canvas, origin="upper")
    ax.set_title(f"{len(curves)} strokes  ({W}×{H} px)")
    ax.axis("off")
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
