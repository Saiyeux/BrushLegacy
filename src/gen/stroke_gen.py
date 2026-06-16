"""
stroke_gen.py — Layer 1: image → painterly brush-stroke data + rendered preview

Approach: grid-based, actual image colours (not palette-quantised).
Each cell gets one stroke coloured by the mean RGB of the source pixels in that cell.
Palette quantisation is deferred to traj_calc.py (robot execution only).

Three layers, large → fine:
  Layer 3  6×6   ≈85 px/cell  — block in main colour areas
  Layer 4  14×14 ≈37 px/cell  — fill colour transitions
  Layer 5  20×20 ≈26 px/cell  — refine edges

Stroke shape: wide near-square ellipse at high alpha → solid colour blocks
that tile together to approximate the original image (PaintTransformer style).
Angle follows local gradient so strokes align with edges.

Outputs:
  data/strokes/{stem}_layer_0{3,4,5}_sorted.csv   — x,y,w,h,θ,r,g,b
  data/output/{stem}_painted.png                   — rendered painting preview

Usage:
    python src/stroke_gen.py --image data/input/Tiger.png
    python src/stroke_gen.py --image data/input/Tiger.png --max_strokes 300
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))         # src/gen/

import cv2
import numpy as np
import pandas as pd

# ── 24-colour palette (used only when exporting robot NPZ, not for rendering) ─
COLOR_CENTERS = [
    (  0,   0,   0), (128, 128, 128), (255, 255, 255),
    (  0, 200,   0), (200,   0,   0), (150,   0, 200),
    (  0, 120, 255), (255, 165,   0), (255, 255,   0), (255, 100, 180),
    (  0, 100,   0), (100,   0,   0), (100,   0, 150),
    (  0,  80, 200), (150,  80,   0), (150, 150,   0), (200,  50, 120),
    (150, 255, 150), (255, 100, 100), (200, 100, 255),
    (100, 180, 255), (255, 200, 100), (255, 255, 150), (255, 150, 200),
]

CANVAS_PX = 512   # rendering canvas and stroke coordinate space

# Per-layer parameters
#   n_grid  : grid dimension (n×n cells)
#   len_r   : stroke length = cell_size × len_r
#   wid_r   : stroke width  = cell_size × wid_r
#   alpha   : blend opacity — high (≥0.90) for solid opaque colour blocks
#   blur    : GaussianBlur kernel for soft edges (applied to rotated-rect mask)
LAYER_CFG = {
    3: dict(n_grid=6,  len_r=1.60, wid_r=0.75, alpha=0.96, blur=5),
    4: dict(n_grid=14, len_r=1.50, wid_r=0.70, alpha=0.92, blur=3),
    5: dict(n_grid=20, len_r=1.40, wid_r=0.65, alpha=0.87, blur=2),
}

COLS = ["x", "y", "w", "h", "θ", "r", "g", "b"]


# ── Colour sampling ───────────────────────────────────────────────────────────

def _cell_color(img_rgb: np.ndarray,
                y1: int, y2: int, x1: int, x2: int) -> tuple[int, int, int]:
    """Mean RGB of the source image patch (actual colour, not palette-snapped)."""
    patch = img_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return (255, 255, 255)
    m = patch.mean(axis=(0, 1))
    return (int(round(m[0])), int(round(m[1])), int(round(m[2])))


def _is_near_white(r: int, g: int, b: int, thr: int = 240) -> bool:
    """True if the cell colour is essentially white canvas — skip it."""
    return r > thr and g > thr and b > thr


# ── Gradient angle ────────────────────────────────────────────────────────────

def _cell_angle(gray_patch: np.ndarray) -> float:
    """Stroke direction perpendicular to dominant local edge (Sobel)."""
    if gray_patch.size < 4:
        return 0.0
    ksize = 3 if min(gray_patch.shape) >= 3 else 1
    gx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=ksize)
    mx, my = float(gx.mean()), float(gy.mean())
    if math.hypot(mx, my) < 0.5:
        return 0.0
    angle = math.atan2(mx, -my)
    angle = angle % math.pi
    if angle > math.pi / 2:
        angle -= math.pi
    return angle


# ── Soft elliptical stroke renderer ──────────────────────────────────────────

def _draw_stroke(canvas: np.ndarray,
                 cx: float, cy: float,
                 length: float, width: float,
                 angle_rad: float,
                 rgb: tuple,
                 alpha: float,
                 blur: int) -> None:
    """Alpha-blend a rotated-rectangle brush stroke onto canvas (RGB, in-place).

    Uses cv2.boxPoints so the stroke looks like a flat painted patch,
    not a circular blob.  A small GaussianBlur softens only the edges.
    """
    H, W = canvas.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    rect = ((cx, cy), (float(length), float(width)), math.degrees(angle_rad))
    box  = cv2.boxPoints(rect)
    box  = np.round(box).astype(np.int32)
    cv2.fillPoly(mask, [box], 255)
    if blur >= 2:
        kk = max(3, blur | 1)
        sigma = blur / 3.0
        mf = cv2.GaussianBlur(mask, (kk, kk), sigma).astype(np.float32) / 255.0
    else:
        mf = mask.astype(np.float32) / 255.0
    w  = mf * alpha
    r, g, b = rgb
    for c, col in enumerate((r, g, b)):
        canvas[:, :, c] = np.clip(
            canvas[:, :, c] * (1.0 - w) + col * w, 0, 255
        ).astype(np.uint8)


# ── Per-layer extraction ──────────────────────────────────────────────────────

def extract_layer(img_rgb: np.ndarray, cfg: dict) -> list[dict]:
    """One stroke per grid cell, coloured by actual mean image colour.

    Returns list of dicts with keys: x, y, w, h, θ, r, g, b.
    Near-white cells are skipped (canvas shows through as highlight).
    """
    n      = cfg["n_grid"]
    H, W   = img_rgb.shape[:2]     # CANVAS_PX × CANVAS_PX
    cell_h = H / n
    cell_w = W / n
    cell   = (cell_h + cell_w) / 2.0
    length = cell * cfg["len_r"]
    width  = cell * cfg["wid_r"]
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    strokes = []
    for gy in range(n):
        for gx in range(n):
            y1 = int(gy * cell_h);  y2 = int((gy + 1) * cell_h)
            x1 = int(gx * cell_w);  x2 = int((gx + 1) * cell_w)
            r, g, b = _cell_color(img_rgb, y1, y2, x1, x2)
            if _is_near_white(r, g, b):
                continue
            cx    = (x1 + x2) / 2.0
            cy    = (y1 + y2) / 2.0
            angle = _cell_angle(gray[y1:y2, x1:x2])
            strokes.append(dict(x=cx, y=cy, w=length, h=width,
                                θ=angle, r=r, g=g, b=b))
    return strokes


# ── Subsampling ───────────────────────────────────────────────────────────────

def _subsample(strokes: list[dict], n: int) -> list[dict]:
    if n <= 0 or len(strokes) <= n:
        return strokes
    idx = np.round(np.linspace(0, len(strokes) - 1, n)).astype(int)
    return [strokes[i] for i in idx]


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def save_layer(strokes: list[dict], path: str) -> int:
    if not strokes:
        return 0
    rows = [[round(s["x"], 1), round(s["y"], 1),
             round(s["w"], 1), round(s["h"], 1),
             round(s["θ"], 6),
             int(s["r"]), int(s["g"]), int(s["b"])]
            for s in strokes]
    df = pd.DataFrame(rows, columns=COLS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(image_path: str, outdir: str, max_strokes: int = 300) -> dict[int, str]:
    t0 = time.perf_counter()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (CANVAS_PX, CANVAS_PX), interpolation=cv2.INTER_AREA)

    raw3 = extract_layer(img_rgb, LAYER_CFG[3])
    raw4 = extract_layer(img_rgb, LAYER_CFG[4])
    raw5 = extract_layer(img_rgb, LAYER_CFG[5])

    s3 = raw3
    s4 = raw4
    s5 = _subsample(raw5, max(0, max_strokes - len(s3) - len(s4)))

    # Base coat: fill canvas with the mean image colour so any gaps between
    # strokes look painted rather than blank white.
    mean_bgr = img_rgb.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    canvas = np.tile(mean_bgr, (CANVAS_PX, CANVAS_PX, 1))
    stem   = Path(image_path).stem
    outdir = Path(outdir)
    paths  = {}

    for lay, strokes in [(3, s3), (4, s4), (5, s5)]:
        cfg = LAYER_CFG[lay]
        for s in strokes:
            _draw_stroke(canvas,
                         s["x"], s["y"], s["w"], s["h"], s["θ"],
                         (s["r"], s["g"], s["b"]),
                         cfg["alpha"], cfg["blur"])
        p = outdir / f"{stem}_layer_{lay:02d}_sorted.csv"
        n = save_layer(strokes, str(p))
        print(f"[stroke_gen] layer {lay}: {n} strokes  ({p.name})")
        if n:
            paths[lay] = str(p)

    out_img = Path(outdir).parent / "output" / f"{stem}_painted.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    ms    = (time.perf_counter() - t0) * 1000
    total = len(s3) + len(s4) + len(s5)
    print(f"[stroke_gen] {total} strokes  ({ms:.0f} ms)")
    print(f"[stroke_gen] painted → {out_img}")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",       required=True)
    p.add_argument("--outdir",      default="data/strokes")
    p.add_argument("--max_strokes", type=int, default=300)
    args = p.parse_args()
    generate(args.image, args.outdir, args.max_strokes)


if __name__ == "__main__":
    main()
