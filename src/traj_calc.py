"""
traj_calc.py  —  8D stroke CSV → Cobrush Pro NPZ (pixel-space curves)

Each 8D stroke (x, y, w, h, θ, r, g, b) in 256-pixel canvas space is
converted to a 2-point pixel-space curve [start, end] at the target canvas
resolution.  Brush width h is saved as width_i for visualization.

NPZ format (Cobrush Pro compatible):
    n_curves      int32
    canvas_width  int32
    canvas_height int32
    curve_i       float32 (2, 2)   [[x_start, y_start], [x_end, y_end]]
    color_i       uint8   (3,)     [r, g, b]
    width_i       float32          brush width in px (optional, for vis)

Usage:
    python src/traj_calc.py \\
        --layer3 data/strokes/Tiger_layer_03_sorted.csv \\
        --layer4 data/strokes/Tiger_layer_04_sorted.csv \\
        --layer5 data/strokes/Tiger_layer_05_sorted.csv \\
        --output data/trajectories/Tiger_curves.npz \\
        --canvas 512 --max_strokes 300
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STROKE_CANVAS = 256   # stroke coords live in [0, 256] (see stroke_convert.py)


def strokes_to_curves(csv_path: str, canvas_px: int,
                      max_strokes: int = 0) -> tuple:
    """Read 8D stroke CSV → (curves, colors, widths).

    Returns:
        curves: list of float32 (2,2) arrays  [[x0,y0],[x1,y1]]
        colors: list of (r,g,b) tuples
        widths: list of float  (brush width in canvas_px space)
    """
    df = pd.read_csv(csv_path)
    required = ["x", "y", "w", "h", "θ", "r", "g", "b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    if max_strokes > 0 and len(df) > max_strokes:
        # Evenly sample to preserve spatial distribution (strokes are grid-sorted)
        idx = np.round(np.linspace(0, len(df) - 1, max_strokes)).astype(int)
        df = df.iloc[idx].reset_index(drop=True)

    scale = canvas_px / STROKE_CANVAS
    curves, colors, widths = [], [], []

    for _, row in df.iterrows():
        cx = float(row["x"]) * scale
        cy = float(row["y"]) * scale
        half_w = float(row["w"]) * scale / 2.0
        brush_w = float(row["h"]) * scale           # minor axis → brush width
        angle = float(row["θ"]) * 2.0               # θ stored halved

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        start = np.array([cx - half_w * cos_a, cy - half_w * sin_a], dtype=np.float32)
        end   = np.array([cx + half_w * cos_a, cy + half_w * sin_a], dtype=np.float32)

        if np.linalg.norm(end - start) < 1.0:
            continue

        curves.append(np.stack([start, end]))
        colors.append((int(row["r"]), int(row["g"]), int(row["b"])))
        widths.append(max(float(brush_w), 1.0))

    return curves, colors, widths


def build_npz(layer_csvs: dict, canvas_px: int, output_path: str,
              max_strokes: int = 300) -> str | None:
    """Merge all layers into one Cobrush Pro NPZ.

    layer_csvs: {3: path, 4: path, 5: path} — processed 3→4→5.
    max_strokes: total cap distributed proportionally across layers.
    """
    # Count available strokes per layer first
    counts = {}
    for layer, path in layer_csvs.items():
        if Path(path).exists():
            df = pd.read_csv(path)
            counts[layer] = len(df)

    if not counts:
        print("[traj_calc] no input files found")
        return None

    # Fixed-weight per-layer allocation: large brush gets most budget
    # Weight: layer3=50%, layer4=35%, layer5=15%  (big strokes dominate appearance)
    LAYER_WEIGHT = {3: 0.50, 4: 0.35, 5: 0.15}
    total_avail = sum(counts.values())
    per_layer: dict[int, int] = {}
    if max_strokes > 0 and total_avail > max_strokes:
        total_weight = sum(LAYER_WEIGHT.get(l, 0.33) for l in counts)
        for layer, n in counts.items():
            w = LAYER_WEIGHT.get(layer, 0.33) / total_weight
            per_layer[layer] = min(n, max(1, round(max_strokes * w)))
        # Adjust rounding error to the heaviest layer
        diff = max_strokes - sum(per_layer.values())
        if diff != 0:
            heaviest = max(counts, key=lambda l: LAYER_WEIGHT.get(l, 0))
            per_layer[heaviest] = max(1, per_layer[heaviest] + diff)
    else:
        per_layer = {layer: 0 for layer in counts}   # 0 = no limit

    all_curves, all_colors, all_widths = [], [], []

    for layer in sorted(layer_csvs):
        path = layer_csvs[layer]
        if not Path(path).exists():
            print(f"[traj_calc] layer {layer}: not found, skipped")
            continue
        lim = per_layer.get(layer, 0)
        curves, colors, widths = strokes_to_curves(path, canvas_px, lim)
        print(f"[traj_calc] layer {layer}: {len(curves)} curves  ({path.name if hasattr(path,'name') else path})")
        all_curves.extend(curves)
        all_colors.extend(colors)
        all_widths.extend(widths)

    if not all_curves:
        print("[traj_calc] no strokes — nothing to save")
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "n_curves":      np.int32(len(all_curves)),
        "canvas_width":  np.int32(canvas_px),
        "canvas_height": np.int32(canvas_px),
    }
    for i, (curve, (r, g, b), w) in enumerate(zip(all_curves, all_colors, all_widths)):
        data[f"curve_{i}"]  = curve.astype(np.float32)
        data[f"color_{i}"]  = np.array([r, g, b], dtype=np.uint8)
        data[f"width_{i}"]  = np.float32(w)

    np.savez(output_path, **data)
    print(f"[traj_calc] {len(all_curves)} total → {output_path}")
    return output_path


def main():
    p = argparse.ArgumentParser(description="8D stroke CSV → Cobrush Pro curves NPZ")
    p.add_argument("--layer3", default=None)
    p.add_argument("--layer4", default=None)
    p.add_argument("--layer5", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--canvas",      type=int, default=512)
    p.add_argument("--max_strokes", type=int, default=300,
                   help="Total stroke cap distributed proportionally (0=no limit)")
    args = p.parse_args()

    layer_csvs = {}
    for layer, path in [(3, args.layer3), (4, args.layer4), (5, args.layer5)]:
        if path:
            layer_csvs[layer] = path

    if not layer_csvs:
        p.error("Provide at least one of --layer3 / --layer4 / --layer5")

    build_npz(layer_csvs, args.canvas, args.output, args.max_strokes)


if __name__ == "__main__":
    main()
