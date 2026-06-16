"""
palette_cfg.py — Palette layout and colour definitions.

Physical layout (3 rows × 8 cols):
     col0      col1  col2  col3    col4    col5  col6   col7
row0: [大红]   [ ]   [ ]   [ ]  [橘红]   [ ]   [ ]    [ ]
row1: [淡黄]  [ ]   [ ]   [ ]  [翠绿]   [ ]   [ ]    [ ]
row2: [湖蓝]  [ ]   [ ]   [ ]  [紫色]   [ ]   [ ]  [黑色]

Slot indices  0=大红  1=橘红  2=淡黄  3=翠绿  4=湖蓝  5=紫色  6=黑色
"""
from __future__ import annotations

# ── Slot definitions ──────────────────────────────────────────────────────────

SLOT_NAMES    = ["大红", "橘红", "淡黄", "翠绿", "湖蓝", "紫色", "黑色"]
PALETTE_NAMES = SLOT_NAMES

# Approximate RGB of the actual pigments on the palette.
# These values drive nearest_slot() — update to match your real paint if needed.
SLOT_RGB = [
    (215,  25,  25),   # 0  大红  — vivid warm red
    (225,  75,  15),   # 1  橘红  — orange-red
    (235, 215,  70),   # 2  淡黄  — pale yellow
    (  0, 165,  65),   # 3  翠绿  — emerald green
    ( 40, 148, 205),   # 4  湖蓝  — lake/cerulean blue
    (135,  10, 170),   # 5  紫色  — purple
    ( 20,  20,  20),   # 6  黑色  — black
]
PALETTE_RGB = SLOT_RGB

# (row, col) position in the 3×8 grid
SLOT_GRID = [
    (0, 0),   # 0  大红
    (0, 4),   # 1  橘红
    (1, 0),   # 2  淡黄
    (1, 4),   # 3  翠绿
    (2, 0),   # 4  湖蓝
    (2, 4),   # 5  紫色
    (2, 7),   # 6  黑色
]

N_SLOTS   = len(SLOT_NAMES)   # 7
REF_SLOT  = 0   # 大红 — primary calibration reference
REF_SLOT2 = 1   # 橘红 — secondary reference (same row, col 4)

DEFAULT_CAL_PATH = "data/calibration/palette.npy"

# ── Grid pitch defaults (metres per column/row unit in the 3×8 grid) ─────────
SLOT_PITCH_X = 0.028   # 28 mm per column unit
SLOT_PITCH_Y = 0.034   # 34 mm per row unit

# ── Action type constants ─────────────────────────────────────────────────────

ACTION_PAINT = 0
ACTION_DIP   = 1
ACTION_WASH  = 2

# ── Colour helpers ────────────────────────────────────────────────────────────

import numpy as np


def _rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """sRGB [0–255] → CIE LAB (D65). No external dependencies."""
    def linearise(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = linearise(r), linearise(g), linearise(b)

    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    X /= 0.95047;  Y /= 1.00000;  Z /= 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(X), f(Y), f(Z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


# Pre-compute palette LAB values once at import time
_SLOT_LAB: list[tuple[float, float, float]] = [_rgb_to_lab(*c) for c in SLOT_RGB]


def nearest_slot(r: int, g: int, b: int) -> int:
    """Return the palette slot index whose colour is perceptually nearest (CIE ΔE76)."""
    qL, qa, qb = _rgb_to_lab(r, g, b)
    best, best_d = 0, float("inf")
    for i, (sL, sa, sb) in enumerate(_SLOT_LAB):
        d = (qL - sL) ** 2 + (qa - sa) ** 2 + (qb - sb) ** 2
        if d < best_d:
            best_d, best = d, i
    return best


def slot_xyz(cal: dict, slot: int, which: str = "dip") -> np.ndarray:
    """Compute robot XYZ for a slot from calibration ref + pitch offset."""
    ref_slot         = int(cal.get("ref_slot", 0))
    ref_xyz          = np.array(cal[f"ref_{which}_xyz"])
    ref_row, ref_col = SLOT_GRID[ref_slot]
    row, col         = SLOT_GRID[slot]
    pitch_x, pitch_y = cal.get("slot_pitch_xy", [SLOT_PITCH_X, SLOT_PITCH_Y])
    return np.array([
        ref_xyz[0] + (col - ref_col) * pitch_x,
        ref_xyz[1] + (row - ref_row) * pitch_y,
        ref_xyz[2],
    ])


def all_slot_positions(cal: dict, which: str = "dip") -> list[np.ndarray]:
    return [slot_xyz(cal, i, which) for i in range(N_SLOTS)]


def save_palette_cal(cal: dict, path: str) -> None:
    np.save(path, cal)
    print(f"  Saved → {path}")
