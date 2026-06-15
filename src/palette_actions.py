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
    Franka, HOME_JOINTS,
    MotionGenerator,
    CartesianVelocities, CartesianVelocitiesFinished,
)
from wash_action import cone_sweep, CONE_N_ROT, CONE_AMP_DEG


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
    spd = speed if speed is not None else _speeds()["hover"]
    robot.go_home(speed_factor=spd)


def _joint_go(robot: Franka, q, speed: float, label: str = "") -> None:
    if label:
        print(f"    [{label}]")
    mg = MotionGenerator(speed, q if isinstance(q, list) else q.tolist())
    robot.robot_control(joint_positions_handle=mg.operator)


def _cart_go(robot: Franka, target_xyz, speed: float, label: str = "") -> None:
    """Cartesian move with exponential velocity smoothing.

    Low-pass filter on velocity commands avoids cartesian_motion_generator
    joint_acceleration_discontinuity reflex on Franka.
    """
    if label:
        print(f"    [{label}]  → [{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}]")
    p_goal = np.array(target_xyz, dtype=np.float64)
    v_cur  = np.zeros(3)
    TAU    = 0.12   # velocity smoothing time constant (s)

    def cb(rs, period):
        dt  = max(period.toSec(), 0.0005)
        T_c = np.array(rs.O_T_EE).reshape(4, 4, order='F')
        err = p_goal - T_c[:3, 3]
        d   = np.linalg.norm(err)

        if d < 0.001 and np.linalg.norm(v_cur) < 0.005:
            return CartesianVelocitiesFinished(CartesianVelocities([0.0] * 6))

        v_des      = (err / d) * min(speed, d * 4.0) if d > 0.001 else np.zeros(3)
        alpha      = 1.0 - math.exp(-dt / TAU)
        v_cur[:]  += alpha * (v_des - v_cur)
        return CartesianVelocities(v_cur.tolist() + [0.0, 0.0, 0.0])

    robot.robot_control(cartesian_velocities_handle=cb)


def _safe_move(robot: Franka, target_xyz, transit_z: float,
               speed: float, label: str = "") -> None:
    """3-stage safe Cartesian move: rise → translate → descend.

    All horizontal travel at transit_z (Hover-2 height) to avoid collisions
    with palette walls and water cup edges.
    """
    target = np.array(target_xyz, dtype=float)
    if label:
        print(f"    [{label}]")

    st  = robot.read_state()
    T_c = np.array(st.O_T_EE).reshape(4, 4, order='F')
    cur = T_c[:3, 3].copy()
    tz  = max(cur[2], transit_z)

    if cur[2] < tz - 0.002:
        _cart_go(robot, np.array([cur[0], cur[1], tz]), speed, "↑ rise")

    mid = np.array([target[0], target[1], tz])
    if np.linalg.norm(mid[:2] - cur[:2]) > 0.002:
        _cart_go(robot, mid, speed, "→ translate")

    if tz - target[2] > 0.002:
        _cart_go(robot, target, speed, "↓ descend")


# ── Atomic actions ────────────────────────────────────────────────────────────

def goto_paint_hover(robot: Franka, cal, slot: int,
                     speed: float | None = None) -> None:
    """Move to Hover-1 above paint slot via transit height (safe 3-stage)."""
    if speed is None:
        speed = _speeds()["hover"]
    T_hov = np.array(cal["slot_hover_T"][slot])
    _safe_move(robot, T_hov[:3, 3], _transit_z(cal), speed,
               label=f"goto hover-1 {_slot_name(slot)}")


def dip_paint(robot: Franka, cal, slot: int,
              speed: float | None = None) -> None:
    """From Hover-1: descend into paint, soak, return to Hover-1. Pure Z motion."""
    spd  = _speeds()
    if speed is None:
        speed = spd["dip"]
    name  = _slot_name(slot)
    T_dip = np.array(cal["slot_dip_T"][slot])
    T_hov = np.array(cal["slot_hover_T"][slot])
    _cart_go(robot, T_dip[:3, 3], speed, f"↓ dip {name}")
    time.sleep(spd["soak_sec"])
    _cart_go(robot, T_hov[:3, 3], speed, f"↑ lift {name}")


def goto_water_hover(robot: Franka, cal,
                     speed: float | None = None) -> None:
    """Move to Hover-2 (transit height above water cup) via safe 3-stage move."""
    if speed is None:
        speed = _speeds()["hover"]
    water_xyz = np.array(cal["water_hover_xyz"])
    _safe_move(robot, water_xyz, _transit_z(cal), speed, label="goto water hover-2")


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
    print(f"    [cone wash]  {n_rot} rot × {amp_deg}°  speed={speed} rad/s  (~{t_rot*n_rot:.1f}s)")
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
    print(f"    [drip wait]  {secs:.1f}s")
    time.sleep(secs)


# ── Compound sequences ────────────────────────────────────────────────────────

def wash_brush(robot: Franka, cal,
               n_rot: int               = CONE_N_ROT,
               amp_deg: float           = CONE_AMP_DEG,
               wash_speed: float | None = None,
               drip_secs: float | None  = None) -> None:
    """Full wash cycle: water hover → dip → sweep → lift → drip."""
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
    print(f"\n  == 换色 → {_slot_name(new_slot)} ==")
    wash_brush(robot, cal, n_rot=n_rot, amp_deg=amp_deg,
               wash_speed=wash_speed, drip_secs=drip_secs)
    goto_paint_hover(robot, cal, new_slot)
    dip_paint(robot, cal, new_slot)
    goto_water_hover(robot, cal)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slot_name(slot: int) -> str:
    from palette_cfg import SLOT_NAMES
    return SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
