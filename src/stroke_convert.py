"""
stroke_convert.py  —  16D PaintTransformer params → 8D (x, y, w, h, θ, r, g, b)

Reads layer_N_strokes.csv produced by inference.py and converts each stroke's
Bezier control points (d0-d7) into a rotated-rectangle representation.

Usage:
    python src/stroke_convert.py \
        --input  data/strokes/layer_03_strokes.csv \
        --output data/strokes/layer_03_8d.csv \
        --layer  3

Render parameters match the original training setup (render_size=80, patch_stride=32)
and are the same for all layers — layer only affects the number of patches.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
PATCH_SIZE   = 32         # PaintTransformer patch stride (pixels)
RENDER_SIZE  = 80         # render window per patch (matches original training setup)
MAX_WIDTH    = 60         # upper bound for stroke length (P0→P3 * render_size)
TARGET_COLS  = ["x", "y", "w", "h", "θ", "r", "g", "b"]


# ── Per-stroke conversion ────────────────────────────────────────────────────

def convert_stroke_16_to_8(p16: np.ndarray,
                             patch_x: int, patch_y: int,
                             render_size: int,
                             patch_size: int = PATCH_SIZE,
                             min_width: float = 0.0,
                             max_width: float = 200.0):
    """Convert one 16D stroke row → (x, y, w, h, θ, r, g, b) or None.

    Stroke geometry uses P0→P3 endpoints (not PCA bounding box) so the
    output rectangle is elongated along the brush direction.
    w = stroke length  (P0→P3 distance, major axis)
    h = brush width    (proportional to d11, minor axis)
    """
    pts = p16[:8].reshape(4, 2).copy()
    if pts.min() < 0:
        pts = (pts + 1.0) / 2.0   # [-1,1] → [0,1]

    p0 = pts[0]            # stroke start (normalised [0,1])
    p3 = pts[3]            # stroke end
    mid = (p0 + p3) / 2.0

    dx = p3[0] - p0[0]
    dy = (1.0 - p3[1]) - (1.0 - p0[1])   # flip Y
    stroke_len = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx))

    # Pixel coordinates of stroke centre
    cx_pix = mid[0] * render_size + patch_x * patch_size
    cy_pix = (1.0 - mid[1]) * render_size + patch_y * patch_size

    # Stroke length (major) and brush width (minor)
    w_pix = stroke_len * render_size          # length along brush direction
    brush_width = float(p16[11])              # d11 encodes brush size [0,1]
    h_pix = max(brush_width * render_size * 0.25, 1.0)   # minor axis

    if w_pix < min_width or w_pix > max_width:
        return None

    # Normalise angle to [-π/2, π/2] (stroke direction, not arrow direction)
    theta = theta % np.pi
    if theta > np.pi / 2:
        theta -= np.pi
    theta_stored = theta / 2.0    # halved so preview can undo with *2

    r = int(np.clip((p16[8]  + 1.0) / 2.0 * 255, 0, 255))
    g = int(np.clip((p16[9]  + 1.0) / 2.0 * 255, 0, 255))
    b = int(np.clip((p16[10] + 1.0) / 2.0 * 255, 0, 255))

    return (cx_pix, cy_pix, w_pix, h_pix, theta_stored, r, g, b)


# ── CSV conversion ───────────────────────────────────────────────────────────

def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder and coerce columns to expected types."""
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[TARGET_COLS].copy()
    for col in ["x", "y", "w", "h", "r", "g", "b"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int)
    df["θ"] = pd.to_numeric(df["θ"], errors="coerce").fillna(0.0).astype(float)
    return df


def convert_csv(input_csv: str, output_csv: str,
                layer: int = 3,
                min_width: float = 0.0) -> str | None:
    """Convert a 16D stroke CSV to 8D and write the result.

    Args:
        input_csv:  path to layer_N_strokes.csv (columns: layer, patch_y, patch_x, stroke_id, d0-d15)
        output_csv: destination path for the 8D CSV
        layer:      painting layer (3, 4, or 5) — controls render size and max_width
        min_width:  minimum stroke width in pixels (default 0 = keep all)

    Returns:
        output_csv on success, None if no valid strokes were produced.
    """
    t0 = time.perf_counter()
    df = pd.read_csv(input_csv, encoding="utf-8")

    required = ["patch_x", "patch_y"] + [f"d{i}" for i in range(16)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    # Canvas size for this layer: (patch_num × PATCH_SIZE) in each axis
    patch_num_x = int(df["patch_x"].max()) + 1
    patch_num_y = int(df["patch_y"].max()) + 1
    canvas_w = patch_num_x * PATCH_SIZE   # e.g. 256 / 512 / 1024
    canvas_h = patch_num_y * PATCH_SIZE
    norm_x = 256.0 / canvas_w
    norm_y = 256.0 / canvas_h

    results = []
    for _, row in df.iterrows():
        p16 = row[[f"d{i}" for i in range(16)]].to_numpy(dtype=np.float32)
        out = convert_stroke_16_to_8(
            p16,
            int(row["patch_x"]), int(row["patch_y"]),
            RENDER_SIZE,
            min_width=min_width, max_width=MAX_WIDTH
        )
        if out is not None:
            cx, cy, w, h, theta, r, g, b = out
            results.append((
                int(cx * norm_x), int(cy * norm_y),
                max(2, int(w * norm_x)), max(2, int(h * norm_y)),
                theta, r, g, b,
            ))

    if not results:
        print(f"[convert] layer {layer}: no valid strokes in {input_csv} — skipping")
        return None

    result_df = pd.DataFrame(results, columns=TARGET_COLS)
    result_df = enforce_types(result_df)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[convert] layer {layer}: {len(result_df)} strokes → {output_csv}  ({elapsed:.1f} ms)")
    return output_csv


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="16D stroke params → 8D rotated-rect CSV")
    p.add_argument("--input",  required=True, help="Input 16D CSV (layer_N_strokes.csv)")
    p.add_argument("--output", required=True, help="Output 8D CSV path")
    p.add_argument("--layer",  type=int, default=3, choices=[3, 4, 5],
                   help="Painting layer (3=large brush, 4=medium, 5=small)")
    p.add_argument("--min_width", type=float, default=0.0,
                   help="Minimum stroke width in pixels (default 0 = keep all)")
    args = p.parse_args()

    result = convert_csv(args.input, args.output, layer=args.layer, min_width=args.min_width)
    if result is None:
        print("[convert] No output written (empty layer).")
        # Exit 0 — empty layer is a warning, not a failure


if __name__ == "__main__":
    main()
