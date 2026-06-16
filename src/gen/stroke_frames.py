"""
stroke_frames.py — Frame-by-frame painting visualization (Layer 1 output)

Produces one PNG per stroke, showing:
  Left panel  — trajectory map: stroke order, transit path, current stroke highlighted
  Right panel — painted canvas accumulating stroke by stroke

Output: data/output/{stem}_frames/frame_NNNN.png   (one per stroke)

Usage:
    python src/stroke_frames.py \
        --layer3 data/strokes/Tiger_layer_03_sorted.csv \
        --layer4 data/strokes/Tiger_layer_04_sorted.csv \
        --layer5 data/strokes/Tiger_layer_05_sorted.csv \
        --outdir data/output/Tiger_frames

    # Faster: skip every N frames (still saves first, last, and layer transitions)
    python src/stroke_frames.py --layer3 ... --step 5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stroke_gen import (COLOR_CENTERS, CANVAS_PX, LAYER_CFG,
                        _draw_stroke, _cell_angle)


# ── Load strokes from CSVs ────────────────────────────────────────────────────

def load_strokes(paths: dict[int, str]) -> list[dict]:
    """Load all layers in order 3→4→5, tag each stroke with its layer."""
    all_strokes = []
    for layer in sorted(paths):
        path = paths[layer]
        if not path or not Path(path).exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            all_strokes.append(dict(
                layer=layer,
                x=float(row["x"]), y=float(row["y"]),
                w=float(row["w"]), h=float(row["h"]),
                θ=float(row["θ"]),
                r=int(row["r"]), g=int(row["g"]), b=int(row["b"]),
            ))
    return all_strokes


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _draw_stroke_on(canvas: np.ndarray, s: dict, layer: int) -> None:
    cfg = LAYER_CFG[layer]
    _draw_stroke(canvas, s["x"], s["y"], s["w"], s["h"], s["θ"],
                 (s["r"], s["g"], s["b"]), cfg["alpha"], cfg["blur"])


def _highlight_stroke(img: np.ndarray, s: dict, color=(0, 220, 0)) -> None:
    """Draw a thin rotated-rectangle outline around the current stroke."""
    rect = ((s["x"], s["y"]), (float(s["w"]), float(s["h"])),
            math.degrees(s["θ"]))
    box  = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(img, [box], True, color, 2, cv2.LINE_AA)
    cv2.circle(img, (int(round(s["x"])), int(round(s["y"]))), 4,
               color, -1, cv2.LINE_AA)


def _draw_traj_map(strokes: list[dict], current: int, size: int) -> np.ndarray:
    """Trajectory map: stroke centres + transit lines, highlight current."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (160, 160, 160), 1)

    # Past transit lines (dashed, grey)
    for i in range(1, current + 1):
        p0 = strokes[i - 1]
        p1 = strokes[i]
        a  = (int(round(p0["x"])), int(round(p0["y"])))
        b  = (int(round(p1["x"])), int(round(p1["y"])))
        dx = b[0] - a[0];  dy = b[1] - a[1]
        dist = max(1, int(math.hypot(dx, dy)))
        for t in range(0, dist, 10):
            t0 = t / dist;  t1 = min((t + 6) / dist, 1.0)
            pa = (int(a[0] + dx * t0), int(a[1] + dy * t0))
            pb = (int(a[0] + dx * t1), int(a[1] + dy * t1))
            cv2.line(img, pa, pb, (195, 195, 195), 1, cv2.LINE_AA)

    # Past stroke outlines (rotated rectangles, coloured)
    for i in range(current + 1):
        s = strokes[i]
        r, g, b = s["r"], s["g"], s["b"]
        col = (max(0, r - 60), max(0, g - 60), max(0, b - 60))
        rect = ((s["x"], s["y"]), (float(s["w"]), float(s["h"])),
                math.degrees(s["θ"]))
        box  = cv2.boxPoints(rect).astype(np.int32)
        lw   = 2 if i == current else 1
        cv2.polylines(img, [box], True, col, lw, cv2.LINE_AA)

    # Current stroke: bright green highlight + index label
    s   = strokes[current]
    cx  = int(round(s["x"]));  cy = int(round(s["y"]))
    cv2.circle(img, (cx, cy), 5, (0, 180, 0), -1, cv2.LINE_AA)
    label = str(current)
    cv2.putText(img, label, (cx + 6, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 140, 0), 1, cv2.LINE_AA)

    return img


def _add_hud(img: np.ndarray, idx: int, total: int, s: dict) -> None:
    """Overlay stroke info at the top of the image."""
    layer_names = {3: "L3 large", 4: "L4 medium", 5: "L5 fine"}
    lay  = s["layer"]
    r, g, b = s["r"], s["g"], s["b"]
    ang  = math.degrees(s["θ"])   # θ key, ASCII-safe label for cv2
    txt  = (f"Stroke {idx + 1:3d}/{total}  {layer_names.get(lay, f'L{lay}')}"
            f"  RGB({r},{g},{b})  ang={ang:.0f}deg")
    H, W = img.shape[:2]
    # dark bar at top
    cv2.rectangle(img, (0, 0), (W, 22), (30, 30, 30), -1)
    cv2.putText(img, txt, (8, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    # colour swatch
    cv2.rectangle(img, (W - 30, 3), (W - 6, 19), (b, g, r), -1)   # BGR for cv2


# ── Main frame generator ──────────────────────────────────────────────────────

def generate_frames(paths: dict[int, str], outdir: Path,
                    step: int = 1, canvas_px: int = CANVAS_PX,
                    image_path: str | None = None) -> None:
    strokes = load_strokes(paths)
    if not strokes:
        print("[frames] no strokes loaded — check CSV paths")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    n = len(strokes)

    # Running painted canvas — start from same base coat as stroke_gen.
    # If the source image is available, use its mean colour; otherwise white.
    if image_path and Path(image_path).exists():
        src = cv2.imread(image_path)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (canvas_px, canvas_px), interpolation=cv2.INTER_AREA)
        mean_bgr = src.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
        paint_canvas = np.tile(mean_bgr, (canvas_px, canvas_px, 1))
    else:
        paint_canvas = np.full((canvas_px, canvas_px, 3), 255, dtype=np.uint8)

    # Layer transition indices (always save these frames regardless of --step)
    layer_starts = set()
    for i, s in enumerate(strokes):
        if i == 0 or strokes[i]["layer"] != strokes[i - 1]["layer"]:
            layer_starts.add(i)

    save_set = (
        set(range(0, n, step))          # every step-th frame
        | layer_starts                   # layer transitions
        | {0, n - 1}                     # first and last
    )

    print(f"[frames] {n} strokes → {len(save_set)} frames  → {outdir}")

    for i, s in enumerate(strokes):
        # Draw onto running canvas
        _draw_stroke_on(paint_canvas, s, s["layer"])

        if i not in save_set:
            continue

        # Left: trajectory map
        traj = _draw_traj_map(strokes, i, canvas_px)

        # Right: painted canvas copy with current stroke highlighted
        paint = paint_canvas.copy()
        _highlight_stroke(paint, s, color=(0, 200, 0))

        # Side-by-side (BGR for cv2.imwrite)
        left_bgr  = cv2.cvtColor(traj,  cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(paint, cv2.COLOR_RGB2BGR)

        # Divider line
        div = np.full((canvas_px, 4, 3), 180, dtype=np.uint8)
        frame = np.hstack([left_bgr, div, right_bgr])

        _add_hud(frame, i, n, s)

        fname = outdir / f"frame_{i:04d}.png"
        cv2.imwrite(str(fname), frame)

    print(f"[frames] done  →  {outdir}/frame_NNNN.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Frame-by-frame painting visualization")
    p.add_argument("--layer3", default=None)
    p.add_argument("--layer4", default=None)
    p.add_argument("--layer5", default=None)
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: data/output/{stem}_frames)")
    p.add_argument("--step",   type=int, default=1,
                   help="Save every N-th frame (default 1 = all frames)")
    p.add_argument("--image", default=None,
                   help="Source image path (for base-coat matching stroke_gen output)")
    args = p.parse_args()

    paths: dict[int, str] = {}
    stem = "painting"
    for layer, path in [(3, args.layer3), (4, args.layer4), (5, args.layer5)]:
        if path and Path(path).exists():
            paths[layer] = path
            stem = Path(path).stem.split("_layer_")[0]

    if not paths:
        p.error("Provide at least one of --layer3 / --layer4 / --layer5")

    outdir = Path(args.outdir) if args.outdir else \
             Path("data/output") / f"{stem}_frames"

    generate_frames(paths, outdir, step=args.step, image_path=args.image)


if __name__ == "__main__":
    main()
