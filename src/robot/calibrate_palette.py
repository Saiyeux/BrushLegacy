"""
calibrate_palette.py — Direct per-slot palette calibration (1 + N_SLOTS + 2 steps).

Workflow
--------
1. Enter descent depth (mm) — how far the brush drops from hover into paint.
For each slot (大红 → 橘红 → 淡黄 → 翠绿 → 湖蓝 → 紫色 → 黑色):
   n. Hand-guide robot to HOVER above that slot → press Enter.
After all slots:
   n+1. Hand-guide robot to HOVER above the water cup → press Enter.
   n+2. Hand-guide robot to DIP into the water → press Enter.

No interpolation or direction vectors — every slot is recorded directly.

Output: data/calibration/palette.npy
  {
    hover_z_offset : float         # metres; dip Z = hover Z − hover_z_offset
    slot_hover_xyz : [[x,y,z], …]  # N_SLOTS entries, one per slot
    water_hover_xyz: [x,y,z]
    water_cup_xyz  : [x,y,z]       # = dip position in water
    water_hover_q  : [7] | None
    water_dip_q    : [7] | None
  }

Usage
-----
  python src/robot/calibrate_palette.py
  python src/robot/calibrate_palette.py --manual
  python src/robot/calibrate_palette.py --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/
sys.path.insert(0, str(Path(__file__).parent))         # src/robot/

import numpy as np

from palette_cfg import (
    PALETTE_RGB, PALETTE_NAMES, N_SLOTS,
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
        print("     Need 3 floats, e.g.  0.520  -0.310  0.175")


# ── Main calibration ──────────────────────────────────────────────────────────

def calibrate(ip: str | None, manual: bool, out_path: Path) -> None:
    total = 1 + N_SLOTS + 2
    print(f"\n{'='*64}")
    print(f"  Palette calibration — direct per-slot  ({total} steps)")
    print(f"  Slots: " + "  ".join(
        f"{_swatch(*PALETTE_RGB[i])}{PALETTE_NAMES[i]}" for i in range(N_SLOTS)))
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
        print("  Manual mode — type XYZ from a measurement tool.\n"
              "  Joint angles will not be recorded.\n")

    if api:
        def record(prompt: str):
            return _record_robot(api, prompt)
    else:
        record = _record_manual

    # ── Step 1: Descent depth ─────────────────────────────────────────────────
    print("  ─── Step 1 / %d: Descent depth ───" % total)
    while True:
        raw = input("  Enter descent depth in mm (hover → dip): ").strip()
        try:
            hover_z_offset = float(raw) / 1000.0
            break
        except ValueError:
            print("  Need a number, e.g.  20")
    print(f"  hover_z_offset = {hover_z_offset * 1000:.1f} mm\n")

    # ── Steps 2 … N+1: Per-slot hover positions ───────────────────────────────
    slot_hover_xyz: list[list[float]] = []
    for i in range(N_SLOTS):
        r, g, b = PALETTE_RGB[i]
        name    = PALETTE_NAMES[i]
        step    = i + 2
        print(f"  ─── Step {step} / {total}: HOVER above {_swatch(r, g, b)} {name} (slot {i}) ───")
        xyz, _ = record(f"Guide to HOVER above {name}.")
        slot_hover_xyz.append(xyz.tolist())
        print()

    # ── Steps N+2 & N+3: Water cup ───────────────────────────────────────────
    print(f"  ─── Step {total - 1} / {total}: HOVER above water cup ───")
    water_hover_xyz, water_hover_q = record("Guide to HOVER above the water cup.")

    print(f"\n  ─── Step {total} / {total}: DIP — brush tip into water ───")
    print("      (tip just touching the water surface at the centre)")
    water_dip_xyz, water_dip_q = record("Guide brush tip INTO the water.")

    # ── Build and save ────────────────────────────────────────────────────────
    cal = {
        "hover_z_offset":  hover_z_offset,
        "slot_hover_xyz":  slot_hover_xyz,
        "water_hover_xyz": water_hover_xyz.tolist(),
        "water_cup_xyz":   water_dip_xyz.tolist(),
        "water_hover_q":   water_hover_q,
        "water_dip_q":     water_dip_q,
    }

    save_palette_cal(cal, str(out_path))
    _print_summary(cal)


# ── Summary display ───────────────────────────────────────────────────────────

def _print_summary(cal: dict) -> None:
    from palette_cfg import slot_xyz as _slot_xyz

    hover_off = float(cal.get("hover_z_offset", 0.02))

    print(f"\n  {'='*64}")
    print(f"  Slot positions  (descent = {hover_off * 1000:.1f} mm)")
    print(f"  {'─'*64}")
    print(f"  {'Slot':>4}  Swatch  {'Name':<10}  "
          f"{'Hover XYZ (m)':<40}  Dip Z (m)")
    print(f"  {'─'*64}")
    for i in range(N_SLOTS):
        r, g, b = PALETTE_RGB[i]
        name    = PALETTE_NAMES[i]
        hover   = _slot_xyz(cal, i, "hover")
        dip_z   = hover[2] - hover_off
        print(f"  {i:4d}   {_swatch(r, g, b)}  {name:<10}  "
              f"{_fmt_xyz(hover):<40}  {dip_z:.4f}")

    water_h = cal["water_hover_xyz"]
    water   = cal["water_cup_xyz"]
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
    print(f"  water_hover_q: {_fmt_q(cal.get('water_hover_q'))}")
    print(f"  water_dip_q:   {_fmt_q(cal.get('water_dip_q'))}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Calibrate palette — direct per-slot hover positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Slots: {' '.join(f'{i}={PALETTE_NAMES[i]}' for i in range(N_SLOTS))}

Steps (total = 1 + {N_SLOTS} slots + 2 water = {1 + N_SLOTS + 2}):
  1.       Enter descent depth in mm
  2–{1+N_SLOTS}.    HOVER above each slot in order
  {2+N_SLOTS}.     HOVER above water cup
  {3+N_SLOTS}.     DIP into water

Examples:
  python src/robot/calibrate_palette.py
  python src/robot/calibrate_palette.py --manual
  python src/robot/calibrate_palette.py --show
""")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from config_loader import robot_ip as _robot_ip
    ip = None if args.manual else _robot_ip()
    calibrate(ip, args.manual, out_path)


if __name__ == "__main__":
    main()
