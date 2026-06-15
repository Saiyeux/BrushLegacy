"""
palette_actions.py — Modular robot actions for palette dipping and washing.

All public functions share the signature (api, cal, ...) where cal is the dict
loaded from data/calibration/palette.npy.

Atomic actions
--------------
goto_paint_hover(api, cal, slot)   → Hover-1 above paint slot
dip_paint(api, cal, slot)          → lower into paint, rise back to Hover-1
goto_water_hover(api, cal)         → Hover-2 above water cup (also transit hub)
dip_water(api, cal)                → lower into water from Hover-2
cone_wash(api, cal, ...)           → conical J5+J6 sweep
drip_wait(cal)                     → time.sleep at Hover-2 for drip

Compound sequences
------------------
wash_brush(api, cal, ...)          → goto_water_hover + dip_water + cone_wash
                                     + goto_water_hover + drip_wait
change_color(api, cal, new_slot)   → wash_brush + goto_paint_hover + dip_paint
"""
from __future__ import annotations

import time
import numpy as np

from wash_action import cone_sweep, CONE_SPEED, CONE_N_ROT, CONE_AMP_DEG, DIP_SPEED, HOVER_SPEED, DRIP_SEC


# ── Low-level motion helpers ──────────────────────────────────────────────────

def _joint_go(api, q, speed, label=""):
    from pyfranka.franka_pybind import MotionGenerator
    if label:
        print(f"    [{label}]")
    mg = MotionGenerator(speed, q if isinstance(q, list) else q.tolist())
    api.robot_control(joint_positions_handle=mg.operator)


def _cart_go(api, target_xyz, ref_T, speed, label=""):
    """P-controller Cartesian move to target_xyz, keeping orientation of ref_T."""
    from pyfranka.franka_pybind import CartesianVelocities, CartesianVelocitiesFinished
    if label:
        print(f"    [{label}]  → {[f'{v:.4f}' for v in target_xyz]}")
    p_goal = np.array(target_xyz, dtype=np.float64)

    def cb(rs, period):
        T_c = np.array(rs.O_T_EE).reshape(4, 4, order='F')
        err = p_goal - T_c[:3, 3]
        d   = np.linalg.norm(err)
        if d < 0.001:
            return CartesianVelocitiesFinished(CartesianVelocities([0.0] * 6))
        v = (err / d) * min(speed, d * 3.0)
        return CartesianVelocities(v.tolist() + [0.0, 0.0, 0.0])

    api.robot_control(cartesian_velocities_handle=cb)


# ── Atomic actions ────────────────────────────────────────────────────────────

def goto_paint_hover(api, cal, slot: int, speed: float = HOVER_SPEED) -> None:
    """Move to Hover-1 position above paint slot."""
    name = _slot_name(slot)
    if slot == 0:   # Red: use recorded joint angles
        _joint_go(api, cal["red_hover_q"], speed, f"goto hover-1 {name}")
    else:
        T = np.array(cal["slot_hover_T"][slot])
        _cart_go(api, T[:3, 3], cal["red_hover_T"], speed, f"goto hover-1 {name}")


def dip_paint(api, cal, slot: int, speed: float = DIP_SPEED) -> None:
    """From Hover-1: descend dip_depth into paint, then return to Hover-1."""
    name   = _slot_name(slot)
    ref_T  = np.array(cal["red_hover_T"])
    T_dip  = np.array(cal["slot_dip_T"][slot])
    T_hov  = np.array(cal["slot_hover_T"][slot])

    _cart_go(api, T_dip[:3, 3], ref_T, speed, f"dip paint {name}")
    time.sleep(0.3)
    _cart_go(api, T_hov[:3, 3], ref_T, speed, f"lift from {name}")


def goto_water_hover(api, cal, speed: float = HOVER_SPEED) -> None:
    """Move to Hover-2 (water cup hover / transit height)."""
    _joint_go(api, cal["water_hover_q"], speed, "goto water hover-2")


def dip_water(api, cal, speed: float = DIP_SPEED) -> None:
    """From Hover-2: descend into water dip position."""
    _joint_go(api, cal["water_dip_q"], speed, "dip into water")


def cone_wash(api, cal,
              n_rot: int     = CONE_N_ROT,
              amp_deg: float = CONE_AMP_DEG,
              speed: float   = CONE_SPEED) -> None:
    """Conical J5+J6 sweep at current (water dip) position."""
    q_dip = np.array(cal["water_dip_q"])
    t_rot = 6.283 / speed
    print(f"    [cone wash]  {n_rot} rot × {amp_deg}°  (~{t_rot*n_rot:.1f}s)")
    cone_sweep(api, q_dip, n_rot=n_rot, amp_deg=amp_deg, speed=speed)


def drip_wait(secs: float = DRIP_SEC) -> None:
    """Wait at Hover-2 for water to drip off brush."""
    print(f"    [drip wait]  {secs:.1f}s")
    time.sleep(secs)


# ── Compound sequences ────────────────────────────────────────────────────────

def wash_brush(api, cal,
               n_rot: int     = CONE_N_ROT,
               amp_deg: float = CONE_AMP_DEG,
               wash_speed: float = CONE_SPEED,
               drip_secs: float  = DRIP_SEC) -> None:
    """Full wash cycle: water hover → dip → sweep → water hover → drip."""
    goto_water_hover(api, cal)
    dip_water(api, cal)
    cone_wash(api, cal, n_rot=n_rot, amp_deg=amp_deg, speed=wash_speed)
    goto_water_hover(api, cal)
    drip_wait(drip_secs)


def change_color(api, cal, new_slot: int,
                 n_rot: int     = CONE_N_ROT,
                 amp_deg: float = CONE_AMP_DEG,
                 wash_speed: float = CONE_SPEED,
                 drip_secs: float  = DRIP_SEC) -> None:
    """Wash brush then dip into new_slot. Call this between color changes."""
    print(f"\n  == 换色 → {_slot_name(new_slot)} ==")
    wash_brush(api, cal, n_rot=n_rot, amp_deg=amp_deg,
               wash_speed=wash_speed, drip_secs=drip_secs)
    goto_paint_hover(api, cal, new_slot)
    dip_paint(api, cal, new_slot)
    goto_water_hover(api, cal)   # return to transit height ready for painting


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slot_name(slot: int) -> str:
    from palette_cfg import SLOT_NAMES
    return SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
