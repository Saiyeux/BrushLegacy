"""
palette_cfg.py — Palette layout, colour mapping, and wash configuration

Physical setup: 3×8 grid, 6 paint colours at every 4th column (col 0 and col 4).
One slot position is calibrated as the reference; all others are computed from
fixed XY pitch increments.

Grid layout (3 rows × 8 cols):
     col0   col1  col2  col3   col4  col5  col6  col7
row0: [R]   [ ]   [ ]   [ ]   [G]   [ ]   [ ]   [ ]
row1: [Y]   [ ]   [ ]   [ ]   [O]   [ ]   [ ]   [ ]
row2: [B]   [ ]   [ ]   [ ]   [P]   [ ]   [ ]   [ ]

Slot indices:
  0=Red  1=Yellow  2=Blue  3=Green  4=Orange  5=Purple

Water cup: separate calibrated position, used for brush washing.
"""

import math
from pathlib import Path

import numpy as np

# ── Palette colour definitions ────────────────────────────────────────────────
# Index 0-5, (R, G, B) in 0-255

PALETTE_RGB = [
    (200,   0,   0),   # 0  Red
    (220, 180,   0),   # 1  Yellow
    (  0,  80, 200),   # 2  Blue
    (  0, 160,   0),   # 3  Green
    (220, 100,   0),   # 4  Orange
    (120,   0, 180),   # 5  Purple
]

PALETTE_NAMES = ["Red", "Yellow", "Blue", "Green", "Orange", "Purple"]

N_SLOTS = len(PALETTE_RGB)   # 6

# ── Grid geometry ─────────────────────────────────────────────────────────────
# (row, col) in the 3×8 physical grid for each slot index.
# Rows increase in the robot's Y direction; cols in X.
SLOT_GRID = [
    (0, 0),   # 0  Red
    (1, 0),   # 1  Yellow
    (2, 0),   # 2  Blue
    (0, 4),   # 3  Green
    (1, 4),   # 4  Orange
    (2, 4),   # 5  Purple
]

# Physical pitch between adjacent grid cells (metres)
SLOT_PITCH_X: float = 0.035   # column pitch (X axis on robot)
SLOT_PITCH_Y: float = 0.035   # row pitch    (Y axis on robot)

# ── Wash configuration ────────────────────────────────────────────────────────
WASH_N_SHAKES:   int   = 5      # number of oscillations in the water cup
WASH_AMPLITUDE:  float = 0.012  # shake amplitude (metres, ±X)


# ── Position helpers ──────────────────────────────────────────────────────────

def slot_xyz(ref_slot: int, ref_xyz, target_slot: int) -> list[float]:
    """Physical XYZ of target_slot given the calibrated position of ref_slot.

    Args:
        ref_slot   : slot index that was physically calibrated.
        ref_xyz    : [x, y, z] calibrated dip position of ref_slot (metres).
        target_slot: slot index whose position we want.

    Returns:
        [x, y, z] dip position of target_slot (metres).
    """
    r0, c0 = SLOT_GRID[ref_slot]
    r1, c1 = SLOT_GRID[target_slot]
    return [
        ref_xyz[0] + (c1 - c0) * SLOT_PITCH_X,
        ref_xyz[1] + (r1 - r0) * SLOT_PITCH_Y,
        float(ref_xyz[2]),
    ]


def all_slot_positions(ref_slot: int, ref_xyz) -> dict[int, list[float]]:
    """Return {slot_idx: [x, y, z]} for all 6 slots."""
    return {i: slot_xyz(ref_slot, ref_xyz, i) for i in range(N_SLOTS)}


# ── Colour mapping ────────────────────────────────────────────────────────────

def nearest_slot(r: int, g: int, b: int) -> int:
    """Return the palette slot index whose colour is nearest to (r, g, b) in RGB space."""
    query = np.array([r, g, b], dtype=float)
    dists = [np.linalg.norm(query - np.array(c)) for c in PALETTE_RGB]
    return int(np.argmin(dists))


def slot_for_strokes(stroke_colors: list[tuple[int, int, int]]) -> list[int]:
    """Map a list of stroke (r, g, b) to palette slot indices."""
    return [nearest_slot(r, g, b) for r, g, b in stroke_colors]


# ── Calibration file I/O ──────────────────────────────────────────────────────

DEFAULT_CAL_PATH = "data/calibration/palette.npy"


def load_palette_cal(path: str = DEFAULT_CAL_PATH) -> dict | None:
    """Load palette calibration saved by calibrate_palette.py.

    Returns dict with keys:
        ref_slot      : int
        ref_dip_xyz   : [x, y, z] in metres
        hover_z       : float  (Z above palette for transit)
        water_cup_xyz : [x, y, z] in metres
        slot_pitch_xy : [dx, dy] in metres (may override defaults)
    Returns None if file not found.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        cal = np.load(str(p), allow_pickle=True).item()
        return cal
    except Exception as e:
        print(f"[palette_cfg] failed to load {path}: {e}")
        return None


def save_palette_cal(cal: dict, path: str = DEFAULT_CAL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, cal)
    print(f"[palette_cfg] saved → {path}")


# ── Action type constants ─────────────────────────────────────────────────────

ACTION_PAINT = 0   # move brush along stroke on canvas
ACTION_DIP   = 1   # dip brush into palette slot
ACTION_WASH  = 2   # wash brush in water cup


def action_name(t: int) -> str:
    return {ACTION_PAINT: "paint", ACTION_DIP: "dip", ACTION_WASH: "wash"}.get(t, f"?{t}")
