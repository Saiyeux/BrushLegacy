"""
traj_calc.py  —  Stroke CSVs → action-sequence NPZ for Cobrush Pro

Each 8D stroke (x, y, w, h, θ, r, g, b) in 512-pixel canvas space is converted
to a 2-point pixel-space curve [start, end].  Palette slot assignment, brush
washing, and dipping are automatically inserted as separate actions.

Action types (see palette_cfg.py):
    0  paint  — move brush along a stroke on the canvas
    1  dip    — dip brush into a palette slot
    2  wash   — wash brush in the water cup

NPZ format (brushlegacy_v2):
    format        string scalar  "brushlegacy_v2"
    n_actions     int32          total number of actions in the sequence
    canvas_width  int32
    canvas_height int32
    action_types  int32[N]       0/1/2 per action
    curves        float32[N,2,2] pixel coords; [0,0,0,0] for non-paint
    colors        uint8[N,3]     actual stroke RGB; palette RGB for dip; zeros for wash
    widths        float32[N]     brush width in px; 0 for non-paint
    slots         int32[N]       palette slot; -1 for wash

    # Palette metadata stored for reference (not needed at execution time if
    # the robot uses its own calibration):
    palette_colors uint8[6,3]    RGB of the 6 palette slots
    palette_names  object(6,)    colour name strings

Usage:
    python src/traj_calc.py \\
        --layer3 data/strokes/Tiger_layer_03_sorted.csv \\
        --layer4 data/strokes/Tiger_layer_04_sorted.csv \\
        --layer5 data/strokes/Tiger_layer_05_sorted.csv \\
        --output data/trajectories/Tiger_actions.npz \\
        --canvas 512 --max_strokes 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))         # src/gen/

import numpy as np
import pandas as pd

from palette_cfg import (
    ACTION_PAINT, ACTION_DIP, ACTION_WASH,
    PALETTE_RGB, PALETTE_NAMES,
    nearest_slot,
)

STROKE_CANVAS = 512   # stroke coordinates live in [0, STROKE_CANVAS]
DIP_INTERVAL  = 10   # re-dip after this many paint strokes with the same colour


# ── CSV → raw stroke list ─────────────────────────────────────────────────────

def _load_csv(csv_path: str, canvas_px: int, max_strokes: int = 0):
    """Read one layer CSV and return (curves, colors, widths) lists.

    curves : list of float32 (2,2) arrays  [[x0,y0],[x1,y1]]
    colors : list of (r,g,b) tuples
    widths : list of float (brush width in canvas_px)
    """
    df = pd.read_csv(csv_path)
    required = ["x", "y", "w", "h", "θ", "r", "g", "b"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    if max_strokes > 0 and len(df) > max_strokes:
        idx = np.round(np.linspace(0, len(df) - 1, max_strokes)).astype(int)
        df  = df.iloc[idx].reset_index(drop=True)

    scale  = canvas_px / STROKE_CANVAS
    curves, colors, widths = [], [], []

    for _, row in df.iterrows():
        cx      = float(row["x"]) * scale
        cy      = float(row["y"]) * scale
        half_w  = float(row["w"]) * scale / 2.0
        brush_w = float(row["h"]) * scale
        angle   = float(row["θ"])

        cos_a = np.cos(angle);  sin_a = np.sin(angle)
        start = np.array([cx - half_w * cos_a, cy - half_w * sin_a], dtype=np.float32)
        end   = np.array([cx + half_w * cos_a, cy + half_w * sin_a], dtype=np.float32)

        if np.linalg.norm(end - start) < 1.0:
            continue

        curves.append(np.stack([start, end]))
        colors.append((int(row["r"]), int(row["g"]), int(row["b"])))
        widths.append(max(float(brush_w), 1.0))

    return curves, colors, widths


# ── Palette slot assignment ───────────────────────────────────────────────────

def _assign_slots(colors: list) -> list[int]:
    """Map each stroke colour to the nearest palette slot index."""
    return [nearest_slot(r, g, b) for r, g, b in colors]


# ── Action sequence builder ───────────────────────────────────────────────────

def _build_action_sequence(curves, colors, widths, slots,
                           dip_interval: int = DIP_INTERVAL) -> tuple:
    """Insert dip, re-dip, and wash actions around paint strokes.

    Rules:
      - Always dip before the very first stroke.
      - When the palette slot changes: wash → dip new slot.
      - Every dip_interval strokes with the same colour: re-dip (no wash).

    Returns (act_types, act_curves, act_colors, act_widths, act_slots) as lists.
    """
    act_types  = []
    act_curves = []
    act_colors = []
    act_widths = []
    act_slots  = []

    def _push(atype, curve=None, color=(0, 0, 0), width=0.0, slot=-1):
        act_types.append(atype)
        act_curves.append(curve if curve is not None
                          else np.zeros((2, 2), dtype=np.float32))
        act_colors.append(color)
        act_widths.append(float(width))
        act_slots.append(int(slot))

    current_slot    = None
    paint_since_dip = 0

    for curve, color, width, slot in zip(curves, colors, widths, slots):
        if slot != current_slot:
            if current_slot is not None:
                _push(ACTION_WASH)
            _push(ACTION_DIP, color=PALETTE_RGB[slot], slot=slot)
            current_slot    = slot
            paint_since_dip = 0
        elif dip_interval > 0 and paint_since_dip >= dip_interval:
            # Re-dip in same colour (no wash)
            _push(ACTION_DIP, color=PALETTE_RGB[slot], slot=slot)
            paint_since_dip = 0

        _push(ACTION_PAINT, curve=curve, color=color, width=width, slot=slot)
        paint_since_dip += 1

    return act_types, act_curves, act_colors, act_widths, act_slots


# ── NPZ builder ───────────────────────────────────────────────────────────────

def build_npz(layer_csvs: dict, canvas_px: int, output_path: str,
              max_strokes: int = 300,
              dip_interval: int = DIP_INTERVAL) -> str | None:
    """Merge all layers into one brushlegacy_v2 action-sequence NPZ.

    layer_csvs  : {3: path, 4: path, 5: path}
    max_strokes : total stroke cap (distributed proportionally)
    """
    # ── Load all layers ──────────────────────────────────────────────────────
    LAYER_WEIGHT = {3: 0.50, 4: 0.35, 5: 0.15}
    counts = {}
    for layer, path in layer_csvs.items():
        if Path(path).exists():
            counts[layer] = len(pd.read_csv(path))

    if not counts:
        print("[traj_calc] no input files found")
        return None

    total_avail = sum(counts.values())
    if max_strokes > 0 and total_avail > max_strokes:
        tw = sum(LAYER_WEIGHT.get(l, 0.33) for l in counts)
        per_layer = {l: min(n, max(1, round(max_strokes * LAYER_WEIGHT.get(l, 0.33) / tw)))
                     for l, n in counts.items()}
        diff = max_strokes - sum(per_layer.values())
        if diff != 0:
            heaviest = max(counts, key=lambda l: LAYER_WEIGHT.get(l, 0))
            per_layer[heaviest] = max(1, per_layer[heaviest] + diff)
    else:
        per_layer = {l: 0 for l in counts}

    all_curves, all_colors, all_widths = [], [], []

    for layer in sorted(layer_csvs):
        path = layer_csvs[layer]
        if not Path(path).exists():
            print(f"[traj_calc] layer {layer}: not found, skipped")
            continue
        c, col, w = _load_csv(path, canvas_px, per_layer.get(layer, 0))
        # Group strokes within each layer by palette slot to minimise washes.
        # Stable sort preserves spatial order within each colour group.
        slots_layer = _assign_slots(col)
        order = sorted(range(len(slots_layer)), key=lambda i: slots_layer[i])
        c   = [c[i]   for i in order]
        col = [col[i] for i in order]
        w   = [w[i]   for i in order]
        # Count unique slots in this layer
        unique = sorted(set(slots_layer))
        from palette_cfg import PALETTE_NAMES
        color_summary = ", ".join(f"{PALETTE_NAMES[s]}×{slots_layer.count(s)}"
                                  for s in unique)
        print(f"[traj_calc] layer {layer}: {len(c)} strokes  [{color_summary}]")
        all_curves.extend(c)
        all_colors.extend(col)
        all_widths.extend(w)

    if not all_curves:
        print("[traj_calc] no strokes — nothing to save")
        return None

    # ── Assign palette slots ─────────────────────────────────────────────────
    all_slots = _assign_slots(all_colors)

    slot_counts = {}
    for s in all_slots:
        slot_counts[s] = slot_counts.get(s, 0) + 1
    print("[traj_calc] palette usage:")
    for s, n in sorted(slot_counts.items()):
        from palette_cfg import PALETTE_NAMES
        print(f"           slot {s} ({PALETTE_NAMES[s]:8s}): {n} strokes")

    # ── Build action sequence ────────────────────────────────────────────────
    types, curves, colors, widths, slots = _build_action_sequence(
        all_curves, all_colors, all_widths, all_slots,
        dip_interval=dip_interval,
    )

    n_paint = sum(1 for t in types if t == ACTION_PAINT)
    n_dip   = sum(1 for t in types if t == ACTION_DIP)
    n_wash  = sum(1 for t in types if t == ACTION_WASH)
    print(f"[traj_calc] action sequence: {len(types)} total"
          f"  ({n_paint} paint, {n_dip} dip, {n_wash} wash)")

    # ── Serialise ────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    N = len(types)
    np.savez(
        output_path,
        format         = "brushlegacy_v2",
        n_actions      = np.int32(N),
        canvas_width   = np.int32(canvas_px),
        canvas_height  = np.int32(canvas_px),
        action_types   = np.array(types,  dtype=np.int32),
        curves         = np.array(curves,  dtype=np.float32),   # N×2×2
        colors         = np.array(colors,  dtype=np.uint8),     # N×3
        widths         = np.array(widths,  dtype=np.float32),   # N
        slots          = np.array(slots,   dtype=np.int32),     # N
        palette_colors = np.array(PALETTE_RGB,   dtype=np.uint8),
        palette_names  = np.array(PALETTE_NAMES, dtype=object),
    )
    print(f"[traj_calc] saved → {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Stroke CSVs → action-sequence NPZ")
    p.add_argument("--layer3", default=None)
    p.add_argument("--layer4", default=None)
    p.add_argument("--layer5", default=None)
    p.add_argument("--output",      required=True)
    p.add_argument("--canvas",       type=int, default=512)
    p.add_argument("--max_strokes",  type=int, default=300,
                   help="Total stroke cap across all layers (0 = no limit)")
    p.add_argument("--dip_interval", type=int, default=DIP_INTERVAL,
                   help=f"Re-dip every N paint strokes, same colour (default {DIP_INTERVAL}; 0=disable)")
    args = p.parse_args()

    layer_csvs = {}
    for layer, path in [(3, args.layer3), (4, args.layer4), (5, args.layer5)]:
        if path:
            layer_csvs[layer] = path

    if not layer_csvs:
        p.error("Provide at least one of --layer3 / --layer4 / --layer5")

    build_npz(layer_csvs, args.canvas, args.output, args.max_strokes,
              dip_interval=args.dip_interval)


if __name__ == "__main__":
    main()
