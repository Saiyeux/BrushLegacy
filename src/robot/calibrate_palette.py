"""
calibrate_palette.py — Record ONE reference palette slot + water cup position.

Physical setup
--------------
  Palette: 3×8 grid, 6 paint colours at every 4th column (col 0 and col 4).

     col0   col1  col2  col3   col4  col5  col6  col7
row0: [R]   [ ]   [ ]   [ ]   [G]   [ ]   [ ]   [ ]
row1: [Y]   [ ]   [ ]   [ ]   [O]   [ ]   [ ]   [ ]
row2: [B]   [ ]   [ ]   [ ]   [P]   [ ]   [ ]   [ ]

  Only ONE slot needs to be physically calibrated.  All other slot positions
  are derived from the row/column pitch (SLOT_PITCH_X, SLOT_PITCH_Y in palette_cfg.py).

  Water cup: a separate container the robot shakes in to clean the brush.

Workflow (6 steps)
------------------
1. Hand-guide robot to HOVER above 大红 (slot 0) → press Enter.
2. Hand-guide robot to DIP into 大红 → press Enter.
3. Hand-guide robot to DIP into 橘红 (slot 1, col direction ref) → press Enter.
4. Hand-guide robot to DIP into 淡黄 (slot 2, row direction ref) → press Enter.
5. Hand-guide robot to HOVER above the water cup → press Enter.
6. Hand-guide robot to DIP into the water → press Enter.

Steps 3 and 4 define the column and row direction vectors in robot space,
so the palette does not need to be axis-aligned.

Output: data/calibration/palette.npy
  {
    ref_slot       : int          # always 0 (大红)
    ref_hover_xyz  : [x,y,z]     # hover above 大红
    ref_dip_xyz    : [x,y,z]     # dip into 大红
    ref_hover_q    : [7]  | None
    ref_dip_q      : [7]  | None
    hover_z_offset : float        # Z(hover) − Z(dip), applied to all slots
    col_vec_xy     : [dx,dy]      # XY displacement per column unit (大红→橘红 / 4)
    row_vec_xy     : [dx,dy]      # XY displacement per row unit   (大红→淡黄 / 1)
    slot_pitch_xy  : [|col|,|row|]# magnitudes, kept for display / backward compat
    water_hover_xyz: [x,y,z]
    water_cup_xyz  : [x,y,z]
    water_hover_q  : [7]  | None
    water_dip_q    : [7]  | None
  }

Usage
-----
  # Robot connected (RT box):
  python src/calibrate_palette.py --ref_slot 0 --ip 192.170.10.200

  # Manual XYZ entry (no robot needed, e.g. on MacBook):
  python src/calibrate_palette.py --ref_slot 0 --manual

  # Custom pitch (default 35mm × 35mm):
  python src/calibrate_palette.py --manual --pitch_x 0.040 --pitch_y 0.035

  # Show a saved calibration:
  python src/calibrate_palette.py --show
  python src/calibrate_palette.py --show data/calibration/palette.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))         # src/robot/

import numpy as np

from palette_cfg import (
    PALETTE_RGB, PALETTE_NAMES, SLOT_GRID, N_SLOTS,
    SLOT_PITCH_X, SLOT_PITCH_Y,
    slot_xyz, all_slot_positions,
    DEFAULT_CAL_PATH,
    save_palette_cal,
)

ROOT = Path(__file__).resolve().parent.parent.parent


# ── Display helpers ───────────────────────────────────────────────────────────

def _swatch(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m   \033[0m"


def _fmt_xyz(xyz) -> str:
    return f"[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]"


def _fmt_q(q) -> str:
    if q is None:
        return "—"
    return "[" + ", ".join(f"{v:.4f}" for v in q) + "]"


# ── Robot interface ───────────────────────────────────────────────────────────

def _connect_robot(ip: str):
    from franka import Franka
    robot = Franka(ip)
    if not robot.wait_ready():
        print("[ABORT] robot not ready")
        sys.exit(1)
    print(f"  Robot connected at {ip}\n")
    return robot


def _read_ee(robot) -> tuple[np.ndarray, list[float]]:
    st = robot.read_state()
    T  = np.array(st.O_T_EE).reshape(4, 4, order='F')
    return T[:3, 3].copy(), list(st.q)


def _record_robot(api, prompt: str) -> tuple[np.ndarray, list[float]]:
    xyz, q = _read_ee(api)
    print(f"     Current EE: {_fmt_xyz(xyz)}")
    input(f"     {prompt}  [Press Enter to record] ")
    xyz, q = _read_ee(api)
    print(f"     Recorded:   {_fmt_xyz(xyz)}")
    return xyz, q


def _record_manual(prompt: str) -> tuple[np.ndarray, None]:
    print(f"     {prompt}  Enter x y z (metres, space-separated):")
    while True:
        parts = input("     x y z → ").strip().split()
        if len(parts) == 3:
            try:
                return np.array([float(p) for p in parts]), None
            except ValueError:
                pass
        print("     Need 3 floats, e.g.  0.720  0.281  0.200")


# ── Main calibration ──────────────────────────────────────────────────────────

_COL_REF_SLOT = 1   # 橘红 — same row as ref (row 0), col 4 → column direction
_ROW_REF_SLOT = 2   # 淡黄 — same col as ref (col 0), row 1 → row direction


def calibrate(ref_slot: int, ip: str | None, manual: bool, out_path: Path) -> None:
    import math

    print(f"\n{'='*64}")
    print(f"  Palette calibration — 3-point reference")
    print(f"  Slots: {PALETTE_NAMES[ref_slot]} (anchor) · "
          f"{PALETTE_NAMES[_COL_REF_SLOT]} (col dir) · "
          f"{PALETTE_NAMES[_ROW_REF_SLOT]} (row dir)")
    print(f"  Output → {out_path}")
    print(f"{'='*64}\n")

    api = None
    if not manual:
        if ip is None:
            print("[ERROR] robot.ip not set in config.yaml — use --manual instead.")
            sys.exit(1)
        api = _connect_robot(ip)
        print("  TIP: Hold the wrist guide button to hand-guide the robot.\n"
              "       Release before pressing Enter to record.\n")
    else:
        print("  Manual mode — type XYZ from a measurement tool or teach pendant.\n"
              "  Joint angles will not be recorded.\n")

    # Unified record() regardless of mode
    if api:
        def record(prompt: str):
            return _record_robot(api, prompt)
    else:
        record = _record_manual

    # ── Steps 1 & 2: Anchor slot (大红) ──────────────────────────────────────
    name = PALETTE_NAMES[ref_slot]
    print(f"  ─── Step 1/6: HOVER above {name} (slot {ref_slot}) ───")
    hover_xyz, hover_q = record(f"Guide to HOVER above {name}.")

    print(f"\n  ─── Step 2/6: DIP into {name} (slot {ref_slot}) ───")
    dip_xyz, dip_q = record(f"Guide to DIP into {name} paint.")

    hover_z_offset = float(hover_xyz[2] - dip_xyz[2])
    print(f"\n  hover_z_offset = {hover_z_offset*1000:.1f} mm  (hover Z − dip Z)")

    # ── Step 3: Column-direction reference (橘红, col 4 same row) ────────────
    col_name = PALETTE_NAMES[_COL_REF_SLOT]
    _, ref_col = SLOT_GRID[ref_slot]
    _, col_col = SLOT_GRID[_COL_REF_SLOT]
    dcol = col_col - ref_col   # = 4

    print(f"\n  ─── Step 3/6: DIP into {col_name} (slot {_COL_REF_SLOT} — col direction) ───")
    col_dip_xyz, _ = record(f"Guide to DIP into {col_name} paint.")

    col_vec_xy = (col_dip_xyz[:2] - dip_xyz[:2]) / dcol

    # ── Step 4: Row-direction reference (淡黄, row 1 same col) ───────────────
    row_name = PALETTE_NAMES[_ROW_REF_SLOT]
    ref_row, _ = SLOT_GRID[ref_slot]
    row_row, _ = SLOT_GRID[_ROW_REF_SLOT]
    drow = row_row - ref_row   # = 1

    print(f"\n  ─── Step 4/6: DIP into {row_name} (slot {_ROW_REF_SLOT} — row direction) ───")
    row_dip_xyz, _ = record(f"Guide to DIP into {row_name} paint.")

    row_vec_xy = (row_dip_xyz[:2] - dip_xyz[:2]) / drow

    print(f"\n  col_vec_xy = [{col_vec_xy[0]*1000:+.2f}, {col_vec_xy[1]*1000:+.2f}] mm/col-unit")
    print(f"  row_vec_xy = [{row_vec_xy[0]*1000:+.2f}, {row_vec_xy[1]*1000:+.2f}] mm/row-unit")

    # ── Steps 5 & 6: Water cup ───────────────────────────────────────────────
    print(f"\n  ─── Step 5/6: HOVER above water cup ───")
    water_hover_xyz, water_hover_q = record("Guide to HOVER above the water cup.")

    print(f"\n  ─── Step 6/6: DIP — brush tip touching water at centre ───")
    print("      (brush vertical or at your normal painting angle,")
    print("       tip just touching the water surface centre)")
    water_dip_xyz, water_dip_q = record("Guide brush tip INTO the water (centre).")

    # ── Build calibration dict ────────────────────────────────────────────────
    cal = {
        "ref_slot":        ref_slot,
        "ref_hover_xyz":   hover_xyz.tolist(),
        "ref_dip_xyz":     dip_xyz.tolist(),
        "ref_hover_q":     hover_q,
        "ref_dip_q":       dip_q,
        "hover_z_offset":  hover_z_offset,
        "col_vec_xy":      col_vec_xy.tolist(),
        "row_vec_xy":      row_vec_xy.tolist(),
        "slot_pitch_xy":   [math.hypot(*col_vec_xy), math.hypot(*row_vec_xy)],
        "water_hover_xyz": water_hover_xyz.tolist(),
        "water_cup_xyz":   water_dip_xyz.tolist(),
        "water_hover_q":   water_hover_q,
        "water_dip_q":     water_dip_q,
    }

    save_palette_cal(cal, str(out_path))
    _print_summary(cal)


def _print_summary(cal: dict) -> None:
    from palette_cfg import slot_xyz as _slot_xyz

    ref       = cal["ref_slot"]
    hover_off = cal["hover_z_offset"]

    if "col_vec_xy" in cal and "row_vec_xy" in cal:
        cv = cal["col_vec_xy"]
        rv = cal["row_vec_xy"]
        vec_info = (f"col_vec=[{cv[0]*1000:+.2f},{cv[1]*1000:+.2f}] mm/col  "
                    f"row_vec=[{rv[0]*1000:+.2f},{rv[1]*1000:+.2f}] mm/row")
    else:
        px, py = cal.get("slot_pitch_xy", [SLOT_PITCH_X, SLOT_PITCH_Y])
        vec_info = f"pitch x={px*1000:.1f} mm  y={py*1000:.1f} mm  (scalar, old format)"

    print(f"\n  {'='*64}")
    print(f"  Computed slot positions  (ref=slot {ref})")
    print(f"  {vec_info}")
    print(f"  {'─'*64}")
    print(f"  {'Slot':>4}  Swatch  {'Name':<10}  "
          f"{'Dip XYZ (m)':<40}  Hover Z (m)")
    print(f"  {'─'*64}")
    for i in range(N_SLOTS):
        r, g, b = PALETTE_RGB[i]
        name    = PALETTE_NAMES[i]
        dip_xyz = _slot_xyz(cal, i, "dip")
        hover_z = dip_xyz[2] + hover_off
        marker  = " ← ref" if i == ref else ""
        print(f"  {i:4d}   {_swatch(r,g,b)}  {name:<10}  "
              f"{_fmt_xyz(dip_xyz):<40}  {hover_z:.4f}{marker}")

    water   = cal["water_cup_xyz"]
    water_h = cal["water_hover_xyz"]
    print(f"\n  Water cup:")
    print(f"    hover: {_fmt_xyz(water_h)}")
    print(f"    dip:   {_fmt_xyz(water)}")
    print(f"  {'='*64}\n")


# ── Show existing calibration ─────────────────────────────────────────────────

def show_cal(path: Path) -> None:
    if not path.exists():
        print(f"[ERROR] Not found: {path}")
        sys.exit(1)
    cal = np.load(str(path), allow_pickle=True).item()
    print(f"\n  Palette calibration: {path}")
    _print_summary(cal)
    print(f"  ref_hover_q:  {_fmt_q(cal.get('ref_hover_q'))}")
    print(f"  ref_dip_q:    {_fmt_q(cal.get('ref_dip_q'))}")
    print(f"  water_hover_q:{_fmt_q(cal.get('water_hover_q'))}")
    print(f"  water_dip_q:  {_fmt_q(cal.get('water_dip_q'))}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Calibrate palette reference slot + water cup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Slot numbering: 0=大红  1=橘红  2=淡黄  3=翠绿  4=湖蓝  5=紫色  6=黑色

Examples:
  # RT box (IP from config.yaml):
  python src/robot/calibrate_palette.py

  # MacBook — manual XYZ entry:
  python src/robot/calibrate_palette.py --manual

  # Review saved calibration:
  python src/robot/calibrate_palette.py --show

Steps (6 total):
  1. HOVER above 大红 (anchor)
  2. DIP  into 大红
  3. DIP  into 橘红 → defines column direction vector
  4. DIP  into 淡黄 → defines row direction vector
  5. HOVER above water cup
  6. DIP  into water (centre)
""")
    p.add_argument("--ref_slot", type=int, default=0,
                   choices=range(N_SLOTS),
                   help="Anchor slot (default 0 = 大红); col/row refs are slots 1 and 2")
    p.add_argument("--manual", action="store_true",
                   help="Enter XYZ manually — no robot connection needed")
    p.add_argument("--out", default=None,
                   help="Output path (default: data/calibration/palette.npy)")
    p.add_argument("--show", nargs="?", const=DEFAULT_CAL_PATH,
                   metavar="FILE",
                   help="Show an existing calibration file and exit")
    args = p.parse_args()

    if args.show is not None:
        show_cal(Path(args.show))
        return

    out_path = Path(args.out) if args.out else ROOT / DEFAULT_CAL_PATH
    from config_loader import robot_ip as _robot_ip
    ip = None if args.manual else _robot_ip()
    calibrate(args.ref_slot, ip, args.manual, out_path)


if __name__ == "__main__":
    main()
