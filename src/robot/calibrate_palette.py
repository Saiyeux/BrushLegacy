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

Workflow
--------
1. Choose which slot to calibrate as the reference (default: slot 0 = Red).
2. Hand-guide robot to HOVER above that slot → press Enter.
3. Hand-guide robot to DIP into the paint → press Enter.
4. Hand-guide robot to HOVER above the water cup → press Enter.
5. Hand-guide robot to DIP into the water → press Enter.
6. Optionally verify by printing all computed slot positions.

Output: data/calibration/palette.npy
  {
    ref_slot      : int          # which slot was physically calibrated
    ref_hover_xyz : [x,y,z]     # hover above reference slot
    ref_dip_xyz   : [x,y,z]     # dip into reference slot
    ref_hover_q   : [7]  | None # joint angles at hover (if robot connected)
    ref_dip_q     : [7]  | None # joint angles at dip
    hover_z_offset: float       # Z difference between hover and dip
    slot_pitch_xy : [dx, dy]    # metres per grid column / row
    water_cup_xyz : [x,y,z]     # dip position inside water cup
    water_hover_xyz: [x,y,z]    # hover above water cup
    water_hover_q : [7]  | None
    water_dip_q   : [7]  | None
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

def calibrate(ref_slot: int, ip: str | None, manual: bool,
              pitch_x: float, pitch_y: float, out_path: Path) -> None:
    r, g, b = PALETTE_RGB[ref_slot]
    name    = PALETTE_NAMES[ref_slot]

    print(f"\n{'='*64}")
    print(f"  Palette calibration — reference slot {ref_slot}: "
          f"{_swatch(r, g, b)} {name}  RGB=({r},{g},{b})")
    print(f"  Grid: 3×8,  col pitch={pitch_x*1000:.1f} mm, "
          f"row pitch={pitch_y*1000:.1f} mm")
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

    record = _record_robot if api else _record_manual

    # ── Step 1 & 2: Reference slot ────────────────────────────────────────────
    print(f"  ─── Step 1/4: HOVER above reference slot ({name}) ───")
    hover_xyz, hover_q = record(api if api else None,
                                f"Guide to HOVER above {name} slot.")

    print(f"\n  ─── Step 2/4: DIP into reference slot ({name}) ───")
    dip_xyz, dip_q = record(api if api else None,
                            f"Guide to DIP into {name} paint.")

    hover_z_offset = float(hover_xyz[2] - dip_xyz[2])
    print(f"\n  hover_z_offset = {hover_z_offset*1000:.1f} mm  "
          f"(hover Z − dip Z)")

    # ── Step 3 & 4: Water cup ─────────────────────────────────────────────────
    # Wash motion is a J5+J6 conical sweep (computed at runtime).
    # Only TWO positions needed: hover above cup, and tip touching water centre.
    print(f"\n  ─── Step 3/4: HOVER above water cup ───")
    water_hover_xyz, water_hover_q = record(api if api else None,
                                            "Guide to HOVER above the water cup.")

    print(f"\n  ─── Step 4/4: DIP — brush tip touching water at centre ───")
    print("      (brush vertical or at your normal painting angle,")
    print("       tip just touching the water surface centre)")
    water_dip_xyz, water_dip_q = record(api if api else None,
                                        "Guide brush tip INTO the water (centre position).")

    # ── Build calibration dict ────────────────────────────────────────────────
    cal = {
        "ref_slot":        ref_slot,
        "ref_hover_xyz":   hover_xyz.tolist(),
        "ref_dip_xyz":     dip_xyz.tolist(),
        "ref_hover_q":     hover_q,
        "ref_dip_q":       dip_q,
        "hover_z_offset":  hover_z_offset,
        "slot_pitch_xy":   [pitch_x, pitch_y],
        "water_hover_xyz": water_hover_xyz.tolist(),
        "water_cup_xyz":   water_dip_xyz.tolist(),
        "water_hover_q":   water_hover_q,
        "water_dip_q":     water_dip_q,
    }

    save_palette_cal(cal, str(out_path))
    _print_summary(cal, pitch_x, pitch_y)


def _print_summary(cal: dict, pitch_x: float = SLOT_PITCH_X,
                   pitch_y: float = SLOT_PITCH_Y) -> None:
    ref       = cal["ref_slot"]
    ref_dip   = cal["ref_dip_xyz"]
    hover_off = cal["hover_z_offset"]

    print(f"\n  {'='*64}")
    print(f"  Computed slot positions  (ref=slot {ref}, "
          f"pitch x={pitch_x*1000:.1f} mm y={pitch_y*1000:.1f} mm)")
    print(f"  {'─'*64}")
    print(f"  {'Slot':>4}  Swatch  {'Name':<10}  "
          f"{'Dip XYZ (m)':<40}  Hover Z (m)")
    print(f"  {'─'*64}")
    for i in range(N_SLOTS):
        r, g, b = PALETTE_RGB[i]
        name    = PALETTE_NAMES[i]
        row, col = SLOT_GRID[i]
        ref_row, ref_col = SLOT_GRID[ref]
        dip_xyz = [
            ref_dip[0] + (col - ref_col) * pitch_x,
            ref_dip[1] + (row - ref_row) * pitch_y,
            ref_dip[2],
        ]
        hover_z = dip_xyz[2] + hover_off
        marker = " ← ref" if i == ref else ""
        print(f"  {i:4d}   {_swatch(r,g,b)}  {name:<10}  "
              f"{_fmt_xyz(dip_xyz):<40}  {hover_z:.4f}{marker}")

    water = cal["water_cup_xyz"]
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
    px, py = cal.get("slot_pitch_xy", [SLOT_PITCH_X, SLOT_PITCH_Y])
    _print_summary(cal, px, py)
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
Slot numbering (3×8 grid, colours at col 0 and col 4):
  0=Red  1=Yellow  2=Blue  3=Green  4=Orange  5=Purple

Examples:
  # RT box — calibrate slot 0 (Red) as reference (IP from config.yaml):
  python src/calibrate_palette.py --ref_slot 0

  # MacBook — manual XYZ entry:
  python src/calibrate_palette.py --ref_slot 0 --manual

  # Review saved calibration:
  python src/calibrate_palette.py --show

Steps recorded (4 total):
  1. HOVER above reference slot
  2. DIP into reference slot
  3. HOVER above water cup
  4. DIP into water (centre)

Wash motion (conical sweep) is computed at runtime from step 4 — no
extra shake positions needed.
""")
    p.add_argument("--ref_slot", type=int, default=0,
                   choices=range(N_SLOTS),
                   help="Which slot to physically calibrate as reference (default 0 = Red)")
    p.add_argument("--manual", action="store_true",
                   help="Enter XYZ manually — no robot connection needed")
    p.add_argument("--pitch_x", type=float, default=SLOT_PITCH_X,
                   help=f"Column pitch in metres (default {SLOT_PITCH_X})")
    p.add_argument("--pitch_y", type=float, default=SLOT_PITCH_Y,
                   help=f"Row pitch in metres (default {SLOT_PITCH_Y})")
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
    calibrate(args.ref_slot, ip, args.manual,
              args.pitch_x, args.pitch_y, out_path)


if __name__ == "__main__":
    main()
