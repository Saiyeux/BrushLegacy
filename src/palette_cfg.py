"""
palette_cfg.py — Palette layout and colour definitions.

Physical layout (3 rows × 8 cols):
     col0     col1  col2  col3   col4    col5  col6   col7
row0: [Red]   [ ]   [ ]   [ ]  [Orange]  [ ]   [ ]    [ ]
row1:[Yellow] [ ]   [ ]   [ ]  [Green]   [ ]   [ ]    [ ]
row2: [Blue]  [ ]   [ ]   [ ]  [Purple]  [ ]   [ ]  [Black]

Slot indices  0=Red  1=Orange  2=Yellow  3=Green  4=Blue  5=Purple  6=Black
"""
from __future__ import annotations

# ── Slot definitions ──────────────────────────────────────────────────────────

SLOT_NAMES   = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Black"]
PALETTE_NAMES = SLOT_NAMES   # alias used by traj_calc / calibrate_palette

SLOT_RGB = [
    (200,   0,   0),   # 0  Red
    (220, 100,   0),   # 1  Orange
    (220, 180,   0),   # 2  Yellow
    (  0, 160,   0),   # 3  Green
    (  0,  80, 200),   # 4  Blue
    (120,   0, 180),   # 5  Purple
    ( 20,  20,  20),   # 6  Black
]
PALETTE_RGB = SLOT_RGB   # alias used by traj_calc / calibrate_palette

# (row, col) position in the 3×8 grid
SLOT_GRID = [
    (0, 0),   # 0  Red
    (0, 4),   # 1  Orange
    (1, 0),   # 2  Yellow
    (1, 4),   # 3  Green
    (2, 0),   # 4  Blue
    (2, 4),   # 5  Purple
    (2, 7),   # 6  Black
]

N_SLOTS   = len(SLOT_NAMES)   # 7
REF_SLOT  = 0   # Red — primary calibration reference
REF_SLOT2 = 1   # Orange — secondary reference (same row, col 4)

DEFAULT_CAL_PATH = "data/calibration/palette.npy"

# ── Grid pitch defaults (metres per column/row unit in the 3×8 grid) ─────────
# col 0 → col 4 = 4 units; row 0 → row 1 = 1 unit
SLOT_PITCH_X = 0.028   # 28 mm per column unit
SLOT_PITCH_Y = 0.034   # 34 mm per row unit

# ── Action type constants ─────────────────────────────────────────────────────

ACTION_PAINT = 0
ACTION_DIP   = 1
ACTION_WASH  = 2

# ── Colour helpers ────────────────────────────────────────────────────────────

import numpy as np


def nearest_slot(r: int, g: int, b: int) -> int:
    q = np.array([r, g, b], dtype=float)
    return int(np.argmin([np.linalg.norm(q - np.array(c)) for c in SLOT_RGB]))


def slot_xyz(cal: dict, slot: int, which: str = "dip") -> np.ndarray:
    """Compute robot XYZ for a slot from calibration ref + pitch offset."""
    ref_slot        = int(cal.get("ref_slot", 0))
    ref_xyz         = np.array(cal[f"ref_{which}_xyz"])
    ref_row, ref_col = SLOT_GRID[ref_slot]
    row, col        = SLOT_GRID[slot]
    pitch_x, pitch_y = cal.get("slot_pitch_xy", [SLOT_PITCH_X, SLOT_PITCH_Y])
    return np.array([
        ref_xyz[0] + (col - ref_col) * pitch_x,
        ref_xyz[1] + (row - ref_row) * pitch_y,
        ref_xyz[2],
    ])


def all_slot_positions(cal: dict, which: str = "dip") -> list[np.ndarray]:
    """Return XYZ positions for all slots."""
    return [slot_xyz(cal, i, which) for i in range(N_SLOTS)]


def save_palette_cal(cal: dict, path: str) -> None:
    np.save(path, cal)
    print(f"  Saved → {path}")
