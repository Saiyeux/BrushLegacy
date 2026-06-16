"""
palette_actions.py — Modular robot actions for palette dipping and washing.

Motion safety
-------------
All horizontal movement happens at Hover-2 height (transit_z = water_hover_xyz[2]).
Every move is decomposed into three Cartesian stages: rise → translate → descend.
This prevents diagonal sweeps that would collide with palette walls or the water cup.

Speeds are read from config.yaml [speeds] section, not hardcoded.

Atomic actions
--------------
go_home(robot)                         → HOME_JOINTS via robot.go_home()
goto_paint_hover(robot, cal, slot)     → Hover-1 above paint slot via transit height
dip_paint(robot, cal, slot)            → lower into paint, rise back (pure Z)
goto_water_hover(robot, cal)           → Hover-2 above water cup via transit height
dip_water(robot, cal)                  → joint move down into water (calibrated)
cone_wash(robot, cal, ...)             → conical J5+J6 sweep at dip position
lift_from_water(robot, cal)            → joint move up back to water_hover_q
drip_wait(secs)                        → time.sleep at Hover-2

Compound sequences
------------------
wash_brush(robot, cal, ...)            → goto_water_hover + dip + sweep + lift + drip
change_color(robot, cal, new_slot)     → wash_brush + goto_paint_hover + dip_paint
"""
from __future__ import annotations

import math
import time
import numpy as np

from franka import (
    Franka, HOME_JOINTS, J7_PIN,
    MotionGenerator,
    CartesianVelocities, CartesianVelocitiesFinished,
)
from wash_action import cone_sweep, CONE_N_ROT, CONE_AMP_DEG
from log import tlog


# ── Config helpers ────────────────────────────────────────────────────────────

def _speeds() -> dict:
    """Load motion speeds from config.yaml with safe fallbacks."""
    try:
        from config_loader import load_config
        s = load_config().get("speeds", {})
    except Exception:
        s = {}
    return {
        "hover":    float(s.get("hover",    0.2)),
        "dip":      float(s.get("dip",      0.05)),
        "cone":     float(s.get("cone",     0.5)),
        "soak_sec": float(s.get("soak_sec", 0.3)),
        "drip_sec": float(s.get("drip_sec", 3.0)),
    }


def _transit_z(cal) -> float:
    return float(np.array(cal["water_hover_xyz"])[2])


# ── Low-level motion helpers ──────────────────────────────────────────────────

def go_home(robot: Franka, speed: float | None = None) -> None:
    """Move to canonical home position."""
    tlog("go_home")
    spd = speed if speed is not None else _speeds()["hover"]
    robot.go_home(speed_factor=spd)


def _joint_go(robot: Franka, q, speed: float, label: str = "") -> None:
    if label:
        tlog(f"  {label}")
    mg = MotionGenerator(speed, q if isinstance(q, list) else q.tolist())
    robot.robot_control(joint_positions_handle=mg.operator)


def _cart_go(robot: Franka, target_xyz, speed: float,
             label: str = "", q7_target: float | None = None) -> None:
    """Cartesian move with exponential velocity smoothing.

    If q7_target is set, uses robot.api.robot_control_j7_pinned so J7 is held
    at that angle via null-space projection throughout the move.
    """
    if label:
        tlog(f"  {label}  → [{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}]")
    p_goal = np.array(target_xyz, dtype=np.float64)
    v_cur  = np.zeros(3)
    TAU    = 0.12

    def cb(rs, period):
        dt  = max(period.toSec(), 0.0005)
        T_c = np.array(rs.O_T_EE).reshape(4, 4, order='F')
        err = p_goal - T_c[:3, 3]
        d   = np.linalg.norm(err)
        if d < 0.001 and np.linalg.norm(v_cur) < 0.005:
            return CartesianVelocitiesFinished(CartesianVelocities([0.0] * 6))
        v_des     = (err / d) * min(speed, d * 4.0) if d > 0.001 else np.zeros(3)
        alpha     = 1.0 - math.exp(-dt / TAU)
        v_cur[:] += alpha * (v_des - v_cur)
        return CartesianVelocities(v_cur.tolist() + [0.0, 0.0, 0.0])

    if q7_target is not None:
        robot.api.robot_control_j7_pinned(cb, q7_target)
    else:
        robot.robot_control(cartesian_velocities_handle=cb)


def _q7(_cal=None) -> float:
    """J7 null-space pin target (fixed robot constant)."""
    return J7_PIN


def _safe_move(robot: Franka, target_xyz, transit_z: float,
               speed: float, label: str = "",
               q7_target: float | None = None) -> None:
    """3-stage safe Cartesian move: rise → translate → descend, J7 pinned."""
    target = np.array(target_xyz, dtype=float)
    if label:
        tlog(f"  {label}")

    st  = robot.read_state()
    T_c = np.array(st.O_T_EE).reshape(4, 4, order='F')
    cur = T_c[:3, 3].copy()
    tz  = max(cur[2], transit_z)

    if cur[2] < tz - 0.002:
        _cart_go(robot, np.array([cur[0], cur[1], tz]), speed, "↑ rise", q7_target)

    mid = np.array([target[0], target[1], tz])
    if np.linalg.norm(mid[:2] - cur[:2]) > 0.002:
        _cart_go(robot, mid, speed, "→ translate", q7_target)

    if tz - target[2] > 0.002:
        _cart_go(robot, target, speed, "↓ descend", q7_target)


# ── Slot position helpers ─────────────────────────────────────────────────────

def _slot_hover_xyz(cal: dict, slot: int) -> np.ndarray:
    """XYZ of hover-1 above palette slot, derived from ref + pitch offset."""
    from palette_cfg import SLOT_GRID
    ref_slot        = int(cal.get("ref_slot", 0))
    ref_xyz         = np.array(cal["ref_hover_xyz"])
    ref_row, ref_col = SLOT_GRID[ref_slot]
    row, col        = SLOT_GRID[slot]
    pitch_x, pitch_y = cal["slot_pitch_xy"]
    return np.array([
        ref_xyz[0] + (col - ref_col) * pitch_x,
        ref_xyz[1] + (row - ref_row) * pitch_y,
        ref_xyz[2],
    ])


def _slot_dip_xyz(cal: dict, slot: int) -> np.ndarray:
    """XYZ of dip position inside palette slot, derived from ref + pitch offset."""
    from palette_cfg import SLOT_GRID
    ref_slot        = int(cal.get("ref_slot", 0))
    ref_xyz         = np.array(cal["ref_dip_xyz"])
    ref_row, ref_col = SLOT_GRID[ref_slot]
    row, col        = SLOT_GRID[slot]
    pitch_x, pitch_y = cal["slot_pitch_xy"]
    return np.array([
        ref_xyz[0] + (col - ref_col) * pitch_x,
        ref_xyz[1] + (row - ref_row) * pitch_y,
        ref_xyz[2],
    ])


# ── Atomic actions ────────────────────────────────────────────────────────────

def goto_paint_hover(robot: Franka, cal, slot: int,
                     speed: float | None = None) -> None:
    """Move to Hover-1 above paint slot. J7 pinned via null-space throughout."""
    if speed is None:
        speed = _speeds()["hover"]
    xyz = _slot_hover_xyz(cal, slot)
    _safe_move(robot, xyz, _transit_z(cal), speed,
               label=f"goto hover-1 {_slot_name(slot)}",
               q7_target=_q7(cal))


def dip_paint(robot: Franka, cal, slot: int,
              speed: float | None = None) -> None:
    """From Hover-1: descend into paint, soak, return to Hover-1. J7 pinned."""
    spd  = _speeds()
    if speed is None:
        speed = spd["dip"]
    name    = _slot_name(slot)
    q7      = _q7(cal)
    xyz_dip = _slot_dip_xyz(cal, slot)
    xyz_hov = _slot_hover_xyz(cal, slot)
    _cart_go(robot, xyz_dip, speed, f"↓ dip {name}", q7)
    tlog(f"  soak {spd['soak_sec']:.2f}s")
    time.sleep(spd["soak_sec"])
    _cart_go(robot, xyz_hov, speed, f"↑ lift {name}", q7)


def goto_water_hover(robot: Franka, cal,
                     speed: float | None = None) -> None:
    """Move to Hover-2 (transit height above water cup). J7 pinned."""
    if speed is None:
        speed = _speeds()["hover"]
    water_xyz = np.array(cal["water_hover_xyz"])
    _safe_move(robot, water_xyz, _transit_z(cal), speed,
               label="goto water hover-2", q7_target=_q7(cal))


def dip_water(robot: Franka, cal, speed: float | None = None) -> None:
    """From Hover-2: joint move down to water dip position (calibrated safe path)."""
    if speed is None:
        speed = _speeds()["dip"]
    _joint_go(robot, cal["water_dip_q"], speed, "↓ dip into water")


def cone_wash(robot: Franka, cal,
              n_rot: int          = CONE_N_ROT,
              amp_deg: float      = CONE_AMP_DEG,
              speed: float | None = None) -> None:
    """Conical J5+J6 sweep at current (water dip) position."""
    if speed is None:
        speed = _speeds()["cone"]
    t_rot = 6.283 / speed
    tlog(f"  cone wash  {n_rot}rot × {amp_deg}°  (~{t_rot * n_rot:.1f}s)")
    cone_sweep(robot, np.array(cal["water_dip_q"]),
               n_rot=n_rot, amp_deg=amp_deg, speed=speed)


def lift_from_water(robot: Franka, cal,
                    speed: float | None = None) -> None:
    """After cone sweep: joint move back to water_hover_q (lift brush out of cup)."""
    if speed is None:
        speed = _speeds()["dip"]
    _joint_go(robot, cal["water_hover_q"], speed, "↑ lift from water")


def drip_wait(secs: float | None = None) -> None:
    """Wait at Hover-2 for water to drip off brush."""
    if secs is None:
        secs = _speeds()["drip_sec"]
    tlog(f"  drip wait {secs:.1f}s")
    time.sleep(secs)


# ── Compound sequences ────────────────────────────────────────────────────────

def wash_brush(robot: Franka, cal,
               n_rot: int               = CONE_N_ROT,
               amp_deg: float           = CONE_AMP_DEG,
               wash_speed: float | None = None,
               drip_secs: float | None  = None) -> None:
    """Full wash cycle: water hover → dip → sweep → lift → drip."""
    tlog("wash_brush")
    goto_water_hover(robot, cal)
    dip_water(robot, cal)
    cone_wash(robot, cal, n_rot=n_rot, amp_deg=amp_deg, speed=wash_speed)
    lift_from_water(robot, cal)
    drip_wait(drip_secs)


def change_color(robot: Franka, cal, new_slot: int,
                 n_rot: int               = CONE_N_ROT,
                 amp_deg: float           = CONE_AMP_DEG,
                 wash_speed: float | None = None,
                 drip_secs: float | None  = None) -> None:
    """Wash brush then dip into new_slot."""
    tlog(f"换色 → {_slot_name(new_slot)}")
    wash_brush(robot, cal, n_rot=n_rot, amp_deg=amp_deg,
               wash_speed=wash_speed, drip_secs=drip_secs)
    goto_paint_hover(robot, cal, new_slot)
    dip_paint(robot, cal, new_slot)
    goto_water_hover(robot, cal)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slot_name(slot: int) -> str:
    from palette_cfg import SLOT_NAMES
    return SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
