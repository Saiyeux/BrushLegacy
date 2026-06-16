"""
stroke_preview.py  —  Render sorted stroke CSVs as a canvas preview image

Draws each stroke as a filled rotated rectangle coloured by its (r, g, b).
Layers are composited in order: 3 (large/background) → 4 → 5 (small/detail).

Usage:
    python src/stroke_preview.py \
        --layer3 data/strokes/painting_layer_03_sorted.csv \
        --layer4 data/strokes/painting_layer_04_sorted.csv \
        --layer5 data/strokes/painting_layer_05_sorted.csv \
        --output data/output/painting_preview.png \
        --size   512
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))         # src/vis/

import cv2
import numpy as np
import pandas as pd


def draw_strokes(csv_path: str, canvas: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Draw one layer's strokes onto canvas in-place. Returns canvas."""
    df = pd.read_csv(csv_path)
    required = ["x", "y", "w", "h", "θ", "r", "g", "b"]
    if not all(c in df.columns for c in required):
        print(f"[preview] skip {csv_path}: missing columns")
        return canvas

    H, W = canvas.shape[:2]
    scale = H / 256.0   # stroke coords live in [0,256]; scale to actual canvas

    for _, row in df.iterrows():
        x, y = float(row["x"]) * scale, float(row["y"]) * scale
        w, h = float(row["w"]) * scale, float(row["h"]) * scale
        theta = float(row["θ"])
        r, g, b = int(row["r"]), int(row["g"]), int(row["b"])

        if w <= 0 or h <= 0:
            continue

        # cv2 uses (centre, (width, height), angle_deg)
        angle_deg = np.degrees(theta * 2)          # undo the /2 from stroke_convert
        rect = ((x, y), (max(w, scale), max(h, scale)), angle_deg)
        pts  = cv2.boxPoints(rect).astype(np.int32)

        # BGR for OpenCV
        color_bgr = (b, g, r)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    return canvas


def render_preview(layer3: str = None,
                   layer4: str = None,
                   layer5: str = None,
                   output: str = "data/output/preview.png",
                   size: int = 512,
                   bg_color: tuple = (255, 255, 255)) -> str:
    """Composite all layers and save preview PNG. Returns output path."""
    canvas = np.full((size, size, 3), bg_color, dtype=np.uint8)

    for path, name in [(layer3, "layer3"), (layer4, "layer4"), (layer5, "layer5")]:
        if path and Path(path).exists():
            draw_strokes(path, canvas)
            df = pd.read_csv(path)
            print(f"[preview] {name}: {len(df)} strokes drawn")
        elif path:
            print(f"[preview] {name}: {path} not found, skipped")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output, canvas)
    print(f"[preview] saved → {output}")
    return output


def main():
    p = argparse.ArgumentParser(description="Render stroke CSVs as canvas preview")
    p.add_argument("--layer3", default=None)
    p.add_argument("--layer4", default=None)
    p.add_argument("--layer5", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--size",   type=int, default=512,
                   help="Canvas size in pixels (default 512)")
    args = p.parse_args()

    render_preview(args.layer3, args.layer4, args.layer5, args.output, args.size)


if __name__ == "__main__":
    main()
