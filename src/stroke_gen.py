"""
stroke_gen.py — Direct image → sorted 8D stroke CSV (no ML inference)

Algorithm:
  For each layer (coarse → fine) we overlay a regular grid on the canvas.
  Each grid cell gets ONE stroke whose:
    • colour  = dominant palette colour in that cell
    • position = cell centre
    • angle   = perpendicular to local image gradient (painterly direction)
    • length  = 1.3 × cell size   (slightly longer than cell for overlap)
    • width   = 0.45 × cell size

  Layer grid sizes (at 256-px canvas):
    layer 3  — 6×6   grid  → 36  strokes, cell≈43 px
    layer 4  — 10×10 grid  → 100 strokes, cell≈26 px
    layer 5  — 14×14 grid  → 196 strokes, cell≈18 px
  Total ≤ 332; --max_strokes cap subsamples each layer proportionally.

Output: {stem}_layer_0{3,4,5}_sorted.csv
        columns: x, y, w, h, θ, r, g, b  — same as stroke_convert.py

Usage:
    python src/stroke_gen.py --image data/input/Tiger.png
    python src/stroke_gen.py --image data/input/Tiger.png --max_strokes 300
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── 24-colour palette (must match stroke_optimize.py) ────────────────────────
COLOR_CENTERS = [
    (  0,   0,   0), (128, 128, 128), (255, 255, 255),
    (  0, 200,   0), (200,   0,   0), (150,   0, 200),
    (  0, 120, 255), (255, 165,   0), (255, 255,   0), (255, 100, 180),
    (  0, 100,   0), (100,   0,   0), (100,   0, 150),
    (  0,  80, 200), (150,  80,   0), (150, 150,   0), (200,  50, 120),
    (150, 255, 150), (255, 100, 100), (200, 100, 255),
    (100, 180, 255), (255, 200, 100), (255, 255, 150), (255, 150, 200),
]

CANVAS = 256   # all stroke coords live in [0, CANVAS] space

# Grid sizes per layer (n × n cells)
# layer 3 = 6×6  = 36  large  cells (≈42 px each at 256-px canvas)
# layer 4 = 12×12 = 144 medium cells (≈21 px each)
# layer 5 = 16×16 = 256 fine   cells (≈16 px each) — subsampled to budget
LAYER_GRIDS = {3: 6, 4: 14, 5: 20}


# ── colour quantisation ───────────────────────────────────────────────────────

def quantize(img_rgb: np.ndarray) -> np.ndarray:
    """Resize to CANVAS×CANVAS; return palette-index map (H×W int)."""
    img  = cv2.resize(img_rgb, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
    flat = img.reshape(-1, 3).astype(np.float32)
    pal  = np.array(COLOR_CENTERS, dtype=np.float32)
    d2   = np.sum((flat[:, None] - pal[None])**2, axis=2)
    return np.argmin(d2, axis=1).reshape(CANVAS, CANVAS)


# ── gradient helpers ──────────────────────────────────────────────────────────

def _cell_angle(gray_patch: np.ndarray) -> float:
    """Stroke angle from local gradient (perpendicular to edge direction)."""
    if gray_patch.size < 4:
        return 0.0
    ksize = 3 if min(gray_patch.shape) >= 3 else 1
    gx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=ksize)
    mx, my = float(gx.mean()), float(gy.mean())
    mag = math.hypot(mx, my)
    if mag < 0.5:           # uniform cell → horizontal stroke
        return 0.0
    # Brush stroke direction is perpendicular to gradient
    angle = math.atan2(mx, -my)
    # Normalise to (−π/2, π/2]
    angle = angle % math.pi
    if angle > math.pi / 2:
        angle -= math.pi
    return angle


# ── grid-based stroke extraction ──────────────────────────────────────────────

def extract_layer(img_rgb: np.ndarray, cidx: np.ndarray,
                  n_grid: int) -> list[dict]:
    """Horizontal strip strokes — merge adjacent same-colour cells in each row.

    Each row is scanned left-to-right; contiguous runs of the same palette
    colour become ONE long stroke (彩条).  This produces far fewer, longer
    strokes than the per-cell approach and avoids the mosaic look.
    """
    H, W = cidx.shape
    cell_h = H / n_grid
    cell_w = W / n_grid

    strokes = []
    for gy in range(n_grid):
        y1 = int(gy * cell_h);  y2 = int((gy + 1) * cell_h)
        cy = (y1 + y2) / 2.0

        # Dominant colour for every cell in this row
        row_ci = []
        for gx in range(n_grid):
            x1c = int(gx * cell_w);  x2c = int((gx + 1) * cell_w)
            patch = cidx[y1:y2, x1c:x2c]
            if patch.size == 0:
                row_ci.append(2)  # treat empty as white (skip)
                continue
            vals, cnts = np.unique(patch.flatten(), return_counts=True)
            row_ci.append(int(vals[np.argmax(cnts)]))

        # Merge adjacent same-colour cells into one horizontal strip
        gx = 0
        while gx < n_grid:
            ci = row_ci[gx]
            if ci == 2:   # pure White — skip
                gx += 1
                continue
            # Find run end
            run_end = gx + 1
            while run_end < n_grid and row_ci[run_end] == ci:
                run_end += 1
            # Strip spans cells [gx, run_end)
            x1s = int(gx * cell_w)
            x2s = int(run_end * cell_w)
            cx = (x1s + x2s) / 2.0
            strip_w = (x2s - x1s) * 1.02   # full horizontal span
            strip_h = cell_h * 1.08          # cell height with slight overlap
            r, g, b = COLOR_CENTERS[ci]
            strokes.append(dict(
                x=cx, y=cy,
                w=strip_w,
                h=strip_h,
                θ=0.0,       # axis-aligned horizontal strip
                r=r, g=g, b=b,
                area=strip_w * strip_h,
            ))
            gx = run_end

    return strokes


# ── subsampling ───────────────────────────────────────────────────────────────

def _subsample(strokes: list[dict], n: int) -> list[dict]:
    """Evenly subsample to n strokes preserving spatial distribution."""
    if n <= 0 or len(strokes) <= n:
        return strokes
    idx = np.round(np.linspace(0, len(strokes) - 1, n)).astype(int)
    return [strokes[i] for i in idx]


# ── write CSV ─────────────────────────────────────────────────────────────────

COLS = ['x', 'y', 'w', 'h', 'θ', 'r', 'g', 'b']


def save_layer(strokes: list[dict], path: str) -> int:
    if not strokes:
        return 0
    rows = [[int(round(s['x'])), int(round(s['y'])),
             max(2, int(round(s['w']))), max(1, int(round(s['h']))),
             float(s['θ']), int(s['r']), int(s['g']), int(s['b'])]
            for s in strokes]
    df = pd.DataFrame(rows, columns=COLS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.6f')
    return len(df)


# ── main ──────────────────────────────────────────────────────────────────────

def generate(image_path: str, outdir: str, max_strokes: int = 300) -> dict[int, str]:
    t0 = time.perf_counter()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cidx = quantize(img_rgb)

    # Budgets: layer3 and layer4 always fill their full grids (complete coverage).
    # Remaining budget goes to layer5 fine details.
    n3 = LAYER_GRIDS[3] ** 2   # 36
    n4 = LAYER_GRIDS[4] ** 2   # 144
    n5 = LAYER_GRIDS[5] ** 2   # 256
    budgets = {
        3: n3,                              # always full
        4: n4,                              # always full
        5: max(0, max_strokes - n3 - n4),   # remainder
    }

    stem   = Path(image_path).stem
    outdir = Path(outdir)
    paths  = {}

    for lay, n_grid in LAYER_GRIDS.items():
        raw = extract_layer(img_rgb, cidx, n_grid)
        sampled = _subsample(raw, budgets[lay])
        p = outdir / f"{stem}_layer_{lay:02d}_sorted.csv"
        n_saved = save_layer(sampled, str(p))
        print(f"[stroke_gen] layer {lay}: {n_saved} strokes  ({p.name})")
        if n_saved:
            paths[lay] = str(p)

    ms = (time.perf_counter() - t0) * 1000
    total = sum(budgets.values())
    print(f"[stroke_gen] {total} total  ({ms:.0f} ms)")
    return paths


def main():
    p = argparse.ArgumentParser(description="Image → sorted stroke CSV (no ML)")
    p.add_argument("--image",       required=True)
    p.add_argument("--outdir",      default="data/strokes")
    p.add_argument("--max_strokes", type=int, default=300)
    args = p.parse_args()
    generate(args.image, args.outdir, args.max_strokes)


if __name__ == "__main__":
    main()
