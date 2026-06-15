"""
palette_cfg.py — Palette layout and colour definitions.

Physical layout (3 rows × 8 cols):
     col0    col1  col2  col3   col4   col5  col6   col7
row0: [Red]  [ ]   [ ]   [ ]  [Yellow] [ ]  [ ]    [ ]
row1:[Orange] [ ]  [ ]   [ ]  [Green]  [ ]  [ ]    [ ]
row2: [Blue]  [ ]  [ ]   [ ]  [Purple] [ ]  [ ]  [Black]

Slot indices  0=Red  1=Yellow  2=Orange  3=Green  4=Blue  5=Purple  6=Black
"""
from __future__ import annotations

# ── Slot definitions ──────────────────────────────────────────────────────────

SLOT_NAMES = ["Red", "Yellow", "Orange", "Green", "Blue", "Purple", "Black"]

SLOT_RGB = [
    (200,   0,   0),   # 0  Red
    (220, 180,   0),   # 1  Yellow
    (220, 100,   0),   # 2  Orange
    (  0, 160,   0),   # 3  Green
    (  0,  80, 200),   # 4  Blue
    (120,   0, 180),   # 5  Purple
    ( 20,  20,  20),   # 6  Black
]

# (row, col) position in the 3×8 grid
SLOT_GRID = [
    (0, 0),   # 0  Red
    (0, 4),   # 1  Yellow
    (1, 0),   # 2  Orange
    (1, 4),   # 3  Green
    (2, 0),   # 4  Blue
    (2, 4),   # 5  Purple
    (2, 7),   # 6  Black
]

N_SLOTS   = len(SLOT_NAMES)   # 7
REF_SLOT  = 0   # Red — primary calibration reference
REF_SLOT2 = 1   # Yellow — secondary reference (determines column direction)

DEFAULT_CAL_PATH = "data/calibration/palette.npy"

# ── Action type constants ─────────────────────────────────────────────────────

ACTION_PAINT = 0
ACTION_DIP   = 1
ACTION_WASH  = 2

# ── Colour helpers ────────────────────────────────────────────────────────────

import numpy as np

def nearest_slot(r: int, g: int, b: int) -> int:
    q = np.array([r, g, b], dtype=float)
    return int(np.argmin([np.linalg.norm(q - np.array(c)) for c in SLOT_RGB]))
