"""
wash_action.py — Conical brush-washing motion using J5 + J6.

Physical principle
------------------
With J1–J4 and J7 fixed, rotating J5 and J6 in a coordinated circular
pattern creates a "绕定点运动" (rotation around a fixed point):
  - J5 provides the cone half-angle (tilt from vertical)
  - J6 provides the azimuthal angle (which direction it tilts)
  - The brush TIP stays approximately at the calibrated water-centre position
  - The brush HANDLE sweeps a cone around that fixed tip

The trajectory for one revolution:
    J5(φ) = J5_0 + A·cos(φ)      φ ∈ [0, 2π)
    J6(φ) = J6_0 + A·sin(φ)
    J1..J4, J7 = unchanged

This is slow (CONE_SPEED) and small amplitude (CONE_AMP_RAD ≈ 4–6°)
so the brush stays well within the water cup.

Calibration needed
------------------
  water_hover_q   — joint config above water cup (transit height)
  water_dip_q     — joint config with brush tip touching water at centre

Usage
-----
    from wash_action import do_wash, cone_trajectory

    # Full wash sequence:
    do_wash(api, cal)                        # default 2 rotations
    do_wash(api, cal, n_rot=3, amp_deg=5)    # 3 rotations, 5° cone

    # Just the waypoints (for preview / custom execution):
    wps = cone_trajectory(q_in_water, n_rot=2, amp_deg=5, steps=32)

Standalone test:
    python src/wash_action.py --cal data/calibration/palette.npy --ip 192.170.10.200
    python src/wash_action.py --cal data/calibration/palette.npy --ip 192.170.10.200 --n 3 --amp 6
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

# ── Parameters ───────────────────────────────────────────────────────────────
J5_IDX = 4          # 0-indexed joint indices for Franka Panda
J6_IDX = 5

CONE_AMP_DEG = 5.0   # default cone half-angle (degrees) — keep small
CONE_N_ROT   = 2     # default number of full rotations
# Speeds come from config.yaml [speeds] section — not hardcoded here


# ── Continuous conical sweep ───────────────────────────────────────────────────

def cone_sweep(api, q_center: np.ndarray,
               n_rot: int     = CONE_N_ROT,
               amp_deg: float = CONE_AMP_DEG,
               speed: float   = 0.5) -> None:
    """Execute conical sweep as a single continuous robot_control call.

    Uses JointVelocities callback — J5 and J6 rotate at `speed` rad/s in
    a coordinated circle. One full rotation takes 2π/speed seconds.
    """
    from pyfranka.franka_pybind import JointVelocities, JointVelocitiesFinished
    amp       = math.radians(amp_deg)
    total_phi = 2.0 * math.pi * n_rot
    phi       = [0.0]

    def callback(robot_state, period):
        phi[0] += speed * period.toSec()
        if phi[0] >= total_phi:
            return JointVelocitiesFinished(JointVelocities([0.0] * 7))
        dq = [0.0] * 7
        dq[J5_IDX] = -amp * speed * math.sin(phi[0])
        dq[J6_IDX] =  amp * speed * math.cos(phi[0])
        return JointVelocities(dq)

    api.robot_control(joint_velocities_handle=callback)


# ── Execution ─────────────────────────────────────────────────────────────────

def do_wash(api, cal: dict,
            n_rot: int     = CONE_N_ROT,
            amp_deg: float = CONE_AMP_DEG,
            speed: float   = CONE_SPEED,
            verbose: bool  = True) -> None:
    """Execute the full brush-washing sequence.

    Sequence:
      1. MotionGenerator → water cup hover
      2. MotionGenerator → dip into water
      3. cone_sweep (continuous JointVelocities)
      4. CartesianPose → lift 3 cm straight up, hold 3 s for drip
      5. MotionGenerator → return to water cup hover
    """
    from pyfranka.franka_pybind import MotionGenerator
    q_hover = np.array(cal["water_hover_q"])
    q_dip   = np.array(cal["water_dip_q"])

    def go(q, spd, label=""):
        if verbose:
            print(f"  [wash] {label}")
        mg = MotionGenerator(spd, q.tolist())
        api.robot_control(joint_positions_handle=mg.operator)

    go(q_hover, HOVER_SPEED, "→ transit to water cup hover")
    go(q_dip,   DIP_SPEED,   "↓ lower brush into water")
    if SOAK_SEC > 0:
        time.sleep(SOAK_SEC)

    if verbose:
        t_rot = 2 * math.pi / speed
        print(f"  [wash] ⊙ cone sweep  {n_rot} rot × {amp_deg}°  "
              f"speed={speed} rad/s  (~{t_rot*n_rot:.1f}s)")
    cone_sweep(api, q_dip, n_rot=n_rot, amp_deg=amp_deg, speed=speed)

    go(q_hover, DIP_SPEED, "↑ lift to water cup hover")
    if verbose:
        print(f"  [wash] ⏳ drip wait {DRIP_SEC} s …")
    time.sleep(DRIP_SEC)

    if verbose:
        print("  [wash] done ✓")


# ── Standalone test ───────────────────────────────────────────────────────────

def _preview(q_center: np.ndarray, n_rot: int, amp_deg: float, steps: int) -> None:
    """Print joint-angle ranges without running the robot."""
    wps = cone_trajectory(q_center, n_rot, amp_deg, steps)
    j5_vals = [w[J5_IDX] for w in wps]
    j6_vals = [w[J6_IDX] for w in wps]
    print(f"\n  Cone trajectory preview  ({len(wps)} waypoints)")
    print(f"  Cone half-angle : {amp_deg:.1f}° = {math.radians(amp_deg):.4f} rad")
    print(f"  J5  range       : [{min(j5_vals):.4f}, {max(j5_vals):.4f}]  "
          f"Δ={max(j5_vals)-min(j5_vals):.4f} rad")
    print(f"  J6  range       : [{min(j6_vals):.4f}, {max(j6_vals):.4f}]  "
          f"Δ={max(j6_vals)-min(j6_vals):.4f} rad")
    print(f"  J1–J4, J7       : unchanged")


def _run_test(ip: str, cal_path: str, n_rot: int, amp_deg: float,
              preview_only: bool) -> None:
    cal_p = Path(cal_path)
    if not cal_p.exists():
        print(f"[ERROR] Not found: {cal_p}")
        return

    cal = np.load(str(cal_p), allow_pickle=True).item()

    for key in ("water_hover_q", "water_dip_q"):
        if cal.get(key) is None:
            print(f"[ERROR] Missing '{key}' in calibration — re-run calibrate_palette.py.")
            return

    q_dip = np.array(cal["water_dip_q"])
    _preview(q_dip, n_rot, amp_deg, CONE_STEPS)

    if preview_only:
        return

    try:
        from pyfranka.franka_pybind import FrankaApi
    except ImportError:
        print("[ERROR] pyfranka not available — use --preview to check trajectory only.")
        return

    print(f"\n  Connecting to {ip} …")
    api = FrankaApi()
    api.init_config(ip, log_size=1000)
    api.set_default_behavior()
    st = api.readOnce()
    if st.robot_mode.name == "kReflex":
        api.automatic_error_recovery()
    print("  Robot ready.\n")

    do_wash(api, cal, n_rot=n_rot, amp_deg=amp_deg)


def main():
    p = argparse.ArgumentParser(
        description="Test / preview conical brush-washing motion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview trajectory (no robot):
  python src/wash_action.py --cal data/calibration/palette.npy --preview

  # Run on robot (IP from config.yaml):
  python src/wash_action.py --cal data/calibration/palette.npy

  # 3 rotations, 6-degree cone:
  python src/wash_action.py --cal data/calibration/palette.npy --n 3 --amp 6
""")
    p.add_argument("--cal",     required=True, help="Calibration file (palette.npy)")
    p.add_argument("--n",       type=int,   default=CONE_N_ROT,   help="Number of rotations")
    p.add_argument("--amp",     type=float, default=CONE_AMP_DEG, help="Cone half-angle (degrees)")
    p.add_argument("--preview", action="store_true",
                   help="Print trajectory ranges only, do not move the robot")
    args = p.parse_args()

    from config_loader import robot_ip
    ip = None if args.preview else robot_ip()
    _run_test(ip, args.cal, args.n, args.amp, args.preview)


if __name__ == "__main__":
    main()
