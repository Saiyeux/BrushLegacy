"""
stroke_optimize.py  —  Filter, colour-snap, and sort 8D strokes for painting

Pipeline:
    1. Remove geometric overlaps (OBB intersection via Separating Axis Theorem)
    2. Remove near-black / near-white / low-saturation strokes
    3. Snap stroke colour to the nearest palette entry
    4. Sort by 16×16 spatial grid then by colour family

Reads a CSV with columns: x, y, w, h, θ, r, g, b
Writes two CSVs: sorted (kept) + removed

Usage:
    python src/stroke_optimize.py \
        --input  data/strokes/layer_03_8d.csv \
        --output data/strokes/layer_03_sorted.csv
"""

import argparse
import csv
import colorsys
import math
import time
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))         # src/gen/
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Palette ──────────────────────────────────────────────────────────────────
COLOR_CENTERS: List[Tuple[int, int, int]] = [
    (0, 0, 0),         (128, 128, 128),  (255, 255, 255),
    (0, 200, 0),       (200, 0, 0),      (150, 0, 200),
    (0, 120, 255),     (255, 165, 0),    (255, 255, 0),    (255, 100, 180),
    (0, 100, 0),       (100, 0, 0),      (100, 0, 150),
    (0, 80, 200),      (150, 80, 0),     (150, 150, 0),    (200, 50, 120),
    (150, 255, 150),   (255, 100, 100),  (200, 100, 255),
    (100, 180, 255),   (255, 200, 100),  (255, 255, 150),  (255, 150, 200),
]
COLOR_NAMES: List[str] = [
    "Black", "Gray", "White",
    "Green", "Red", "Purple", "Blue", "Orange", "Yellow", "Pink",
    "DarkGreen", "DarkRed", "DarkPurple", "DarkBlue", "DarkOrange",
    "DarkYellow", "DarkPink",
    "LightGreen", "LightRed", "LightPurple", "LightBlue",
    "LightOrange", "LightYellow", "LightPink",
]
COLOR_ORDER: Dict[str, int] = {name: i for i, name in enumerate(COLOR_NAMES)}
NAME_TO_RGB: Dict[str, Tuple[int, int, int]] = dict(zip(COLOR_NAMES, COLOR_CENTERS))
RGB_TO_NAME: Dict[Tuple[int, int, int], str] = {v: k for k, v in NAME_TO_RGB.items()}

# ── Thresholds ────────────────────────────────────────────────────────────────
MAX_COLOR_DIST  = 180.0
MIN_SATURATION  = 30        # low: dark/brown/grey strokes still pass (tiger stripes)
BLACK_THRESH    = 0.05      # r,g,b ALL below 12 → discard (pure black paint gap)
WHITE_THRESH    = 252       # r,g,b ALL above this → discard (bare canvas)
OVERLAP_RATIO   = 0.70      # SAT overlap ratio to call two strokes conflicting
MIN_OVERLAP_AREA = 50.0     # minimum area (px²) to bother checking

GRID_NX = 16
GRID_NY = 16


# ── OBB via Separating Axis Theorem ──────────────────────────────────────────

class RotatedRect:
    """Oriented bounding box in 2D."""

    def __init__(self, x: float, y: float,
                 w: float, h: float, theta: float):
        self.cx    = x
        self.cy    = y
        self.half_w = abs(w) / 2.0
        self.half_h = abs(h) / 2.0
        cos_a = math.cos(theta)
        sin_a = math.sin(theta)
        self.axes  = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        self.area  = abs(w) * abs(h)

    def vertices(self) -> np.ndarray:
        hw, hh = self.half_w, self.half_h
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        return corners @ self.axes + np.array([self.cx, self.cy])

    def aabb(self) -> Tuple[float, float, float, float]:
        v = self.vertices()
        return float(v[:, 0].min()), float(v[:, 1].min()), \
               float(v[:, 0].max()), float(v[:, 1].max())


def _project_onto(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    projs = vertices @ axis
    return float(projs.min()), float(projs.max())


def obb_intersection_area(a: RotatedRect, b: RotatedRect) -> float:
    """Approximate OBB intersection area using SAT overlap lengths product.

    Not exact for OBBs but fast and sufficient for overlap filtering.
    Returns 0 if the two boxes do not overlap on any SAT axis.
    """
    va, vb = a.vertices(), b.vertices()
    axes = list(a.axes) + list(b.axes)
    overlaps = []
    for ax in axes:
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        a_min, a_max = _project_onto(va, ax)
        b_min, b_max = _project_onto(vb, ax)
        ov = min(a_max, b_max) - max(a_min, b_min)
        if ov <= 0:
            return 0.0
        overlaps.append(ov)
    return float(overlaps[0] * overlaps[1])


def aabb_overlaps(a: RotatedRect, b: RotatedRect) -> bool:
    ax1, ay1, ax2, ay2 = a.aabb()
    bx1, by1, bx2, by2 = b.aabb()
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


# ── Colour utilities ─────────────────────────────────────────────────────────

def _color_dist(c1, c2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def _saturation(rgb) -> float:
    r, g, b = rgb
    mx, mn = max(r, g, b), min(r, g, b)
    return 0.0 if mx == 0 else (mx - mn) / mx * 255.0


def _hsv(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360, s * 100, v * 100


def _color_category(r, g, b) -> str:
    h, s, v = _hsv(r, g, b)
    if s < 10:
        return "White" if v > 90 else ("Black" if v < 15 else "Gray")
    if h >= 300 or h < 5:   hue = "Red"
    elif h < 35:             hue = "Orange"
    elif h < 75:             hue = "Yellow"
    elif h < 165:            hue = "Green"
    elif h < 240:            hue = "Blue"
    elif h < 300:            hue = "Purple"
    else:                    hue = "Pink"
    light = "Light" if v > 65 else ("Dark" if v < 30 else "")
    return f"{light}{hue}" if light else hue


# ── Step 1: geometric overlap removal ────────────────────────────────────────

def remove_overlaps(strokes: List[List[float]]):
    if not strokes:
        return [], []
    rects = [RotatedRect(*s[:5]) for s in strokes]

    # Skip overlap removal when all strokes are tiny (saves O(n²) on layer 5)
    max_area = max(r.area for r in rects)
    if max_area < MIN_OVERLAP_AREA:
        return list(strokes), []

    by_area = sorted(range(len(strokes)), key=lambda i: rects[i].area, reverse=True)
    kept: List[int] = []
    removed: List[int] = []
    for i in by_area:
        ri = rects[i]
        if ri.area <= 0:
            removed.append(i)
            continue
        if ri.area < MIN_OVERLAP_AREA:
            # Tiny stroke: can't meaningfully overlap anything, just keep it
            kept.append(i)
            continue
        conflict = False
        for k in kept:
            if rects[k].area < MIN_OVERLAP_AREA:
                continue
            if not aabb_overlaps(ri, rects[k]):
                continue
            inter = obb_intersection_area(ri, rects[k])
            if inter > MIN_OVERLAP_AREA:
                ratio = inter / min(ri.area, rects[k].area)
                if ratio > OVERLAP_RATIO:
                    conflict = True
                    break
        (removed if conflict else kept).append(i)
    kept_sorted    = [strokes[i] for i in sorted(kept)]
    removed_sorted = [strokes[i] for i in sorted(removed)]
    return kept_sorted, removed_sorted


# ── Step 2: colour snap + saturation filter ──────────────────────────────────

def classify_and_snap(strokes: List[List[float]]):
    kept, removed = [], []
    for s in strokes:
        r, g, b = float(s[5]), float(s[6]), float(s[7])
        if (r < BLACK_THRESH * 255 and g < BLACK_THRESH * 255 and b < BLACK_THRESH * 255) or \
           (r > WHITE_THRESH and g > WHITE_THRESH and b > WHITE_THRESH):
            removed.append(s)
            continue
        if _saturation((r, g, b)) < MIN_SATURATION:
            removed.append(s)
            continue
        # Nearest-distance snap to palette (avoids category mis-mapping)
        dists = [_color_dist((r, g, b), c) for c in COLOR_CENTERS]
        idx   = int(np.argmin(dists))
        if dists[idx] > MAX_COLOR_DIST:
            removed.append(s)
            continue
        sr, sg, sb = COLOR_CENTERS[idx]
        ns = s.copy()
        ns[5], ns[6], ns[7] = sr, sg, sb
        kept.append(ns)
    return kept, removed


# ── Step 3: 16×16 grid sort ──────────────────────────────────────────────────

def _dominant_cell(rect: RotatedRect, canvas: int = 256) -> Tuple[int, int]:
    x1, y1, x2, y2 = rect.aabb()
    areas: DefaultDict[Tuple[int, int], float] = defaultdict(float)
    for gx in range(max(0, int(x1 / canvas * GRID_NX)),
                    min(GRID_NX, int(x2 / canvas * GRID_NX) + 1)):
        for gy in range(max(0, int(y1 / canvas * GRID_NY)),
                        min(GRID_NY, int(y2 / canvas * GRID_NY) + 1)):
            cx1 = gx * canvas / GRID_NX
            cy1 = gy * canvas / GRID_NY
            cx2 = cx1 + canvas / GRID_NX
            cy2 = cy1 + canvas / GRID_NY
            ix  = (min(x2, cx2) - max(x1, cx1))
            iy  = (min(y2, cy2) - max(y1, cy1))
            if ix > 0 and iy > 0:
                areas[(gx, gy)] = ix * iy
    if areas:
        return max(areas.items(), key=lambda kv: kv[1])[0]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (max(0, min(GRID_NX - 1, int(cx / canvas * GRID_NX))),
            max(0, min(GRID_NY - 1, int(cy / canvas * GRID_NY))))


def grid_layer_sort(strokes: List[List[float]]) -> List[List[float]]:
    if not strokes:
        return strokes
    rects  = [RotatedRect(*s[:5]) for s in strokes]
    colors = [RGB_TO_NAME.get((int(s[5]), int(s[6]), int(s[7])), "Gray") for s in strokes]

    grid: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, s in enumerate(strokes):
        gx, gy = _dominant_cell(rects[i])
        grid[(gx, gy)].append(i)

    max_layers = max((len(v) for v in grid.values()), default=0)
    final: List[int] = []
    for layer_idx in range(max_layers):
        layer_strokes = []
        for gy in range(GRID_NY):
            for gx in range(GRID_NX):
                key = (gx, gy)
                if key in grid and layer_idx < len(grid[key]):
                    layer_strokes.append(grid[key][layer_idx])
        layer_strokes.sort(key=lambda i: COLOR_ORDER.get(colors[i], 10_000))
        final.extend(layer_strokes)
    return [strokes[i] for i in final]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def optimize(input_csv: str, output_csv: str,
             removed_csv: Optional[str] = None) -> None:
    t0 = time.perf_counter()

    df = pd.read_csv(input_csv)
    req = ["x", "y", "w", "h", "θ", "r", "g", "b"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    for col in req:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=req)
    for col in ["r", "g", "b"]:
        df[col] = df[col].clip(0, 255)

    strokes = df.values.tolist()
    print(f"[optimize] input: {len(strokes)} strokes from {input_csv}")

    kept, rm1 = remove_overlaps(strokes)
    print(f"[optimize] after overlap removal: {len(kept)} kept, {len(rm1)} removed")

    kept, rm2 = classify_and_snap(kept)
    print(f"[optimize] after colour snap: {len(kept)} kept, {len(rm2)} removed")

    kept = grid_layer_sort(kept)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(req)
        w.writerows(kept)

    removed_all = rm1 + rm2
    if removed_csv and removed_all:
        Path(removed_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(removed_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(req)
            w.writerows(removed_all)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[optimize] {len(kept)} strokes → {output_csv}  ({elapsed:.1f} ms)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Filter overlaps, snap colours, and sort 8D stroke CSV")
    p.add_argument("--input",   required=True, help="8D stroke CSV (x,y,w,h,θ,r,g,b)")
    p.add_argument("--output",  required=True, help="Output sorted CSV path")
    p.add_argument("--removed", default=None,  help="Optional path to write removed strokes")
    args = p.parse_args()

    optimize(args.input, args.output, args.removed)


if __name__ == "__main__":
    main()
