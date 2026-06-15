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
goto_paint_hover(api, cal, slot)   → Hover-1 above paint slot via transit height
dip_paint(api, cal, slot)          → lower into paint, rise back (pure Z)
goto_water_hover(api, cal)         → Hover-2 above water cup via transit height
dip_water(api, cal)                → joint move down into water (calibrated)
cone_wash(api, cal, ...)           → conical J5+J6 sweep at dip position
lift_from_water(api, cal)          → joint move up back to water_hover_q
drip_wait(secs)                    → time.sleep at Hover-2

Compound sequences
------------------
wash_brush(api, cal, ...)          → goto_water_hover + dip + sweep + lift + drip
change_color(api, cal, new_slot)   → wash_brush + goto_paint_hover + dip_paint
"""
from __future__ import annotations

import time
import numpy as np

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
    """Transit height = Hover-2 Z (water cup hover); all horizontal moves at this Z."""
    return float(np.array(cal["water_hover_xyz"])[2])


# ── Low-level motion helpers ──────────────────────────────────────────────────

def _joint_go(api, q, speed: float, label: str = "") -> None:
    from pyfranka.franka_pybind import MotionGenerator
    if label:
        print(f"    [{label}]")
    mg = MotionGenerator(speed, q if isinstance(q, list) else q.tolist())
    api.robot_control(joint_positions_handle=mg.operator)


def _cart_go(api, target_xyz, speed: float, label: str = "") -> None:
    """P-controller Cartesian move to target_xyz; stops within 1 mm."""
    from pyfranka.franka_pybind import CartesianVelocities, CartesianVelocitiesFinished
    if label:
        print(f"    [{label}]  → [{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}]")
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


def _safe_move(api, target_xyz, transit_z: float, speed: float, label: str = "") -> None:
    """3-stage safe Cartesian move: rise → translate → descend.

    Phase 1: rise  — move Z up to transit_z (keep XY)
    Phase 2: translate — move XY to target XY (stay at transit_z)
    Phase 3: descend — move Z down to target Z (keep XY)

    If already above transit_z, uses current Z as transit height.
    Never moves diagonally — eliminates collision risk with palette/water cup.
    """
    target = np.array(target_xyz, dtype=float)
    if label:
        print(f"    [{label}]")

    st  = api.readOnce()
    T_c = np.array(st.O_T_EE).reshape(4, 4, order='F')
    cur = T_c[:3, 3].copy()
    tz  = max(cur[2], transit_z)    # if already higher, stay high

    # Phase 1: rise
    if cur[2] < tz - 0.002:
        _cart_go(api, np.array([cur[0], cur[1], tz]), speed, "↑ rise")

    # Phase 2: translate XY
    mid = np.array([target[0], target[1], tz])
    if np.linalg.norm(mid[:2] - cur[:2]) > 0.002:
        _cart_go(api, mid, speed, "→ translate")

    # Phase 3: descend
    if tz - target[2] > 0.002:
        _cart_go(api, target, speed, "↓ descend")


# ── Atomic actions ────────────────────────────────────────────────────────────

def goto_paint_hover(api, cal, slot: int, speed: float | None = None) -> None:
    """Move to Hover-1 above paint slot via transit height (safe 3-stage)."""
    if speed is None:
        speed = _speeds()["hover"]
    T_hov = np.array(cal["slot_hover_T"][slot])
    _safe_move(api, T_hov[:3, 3], _transit_z(cal), speed,
               label=f"goto hover-1 {_slot_name(slot)}")


def dip_paint(api, cal, slot: int, speed: float | None = None) -> None:
    """From Hover-1: descend into paint, soak, return to Hover-1. Pure Z motion."""
    spd   = _speeds()
    if speed is None:
        speed = spd["dip"]
    name  = _slot_name(slot)
    T_dip = np.array(cal["slot_dip_T"][slot])
    T_hov = np.array(cal["slot_hover_T"][slot])
    _cart_go(api, T_dip[:3, 3], speed, f"↓ dip {name}")
    time.sleep(spd["soak_sec"])
    _cart_go(api, T_hov[:3, 3], speed, f"↑ lift {name}")


def goto_water_hover(api, cal, speed: float | None = None) -> None:
    """Move to Hover-2 (transit height above water cup) via safe 3-stage move."""
    if speed is None:
        speed = _speeds()["hover"]
    water_xyz = np.array(cal["water_hover_xyz"])
    _safe_move(api, water_xyz, _transit_z(cal), speed, label="goto water hover-2")


def dip_water(api, cal, speed: float | None = None) -> None:
    """From Hover-2: joint move down to water dip position (calibrated safe path)."""
    if speed is None:
        speed = _speeds()["dip"]
    _joint_go(api, cal["water_dip_q"], speed, "↓ dip into water")


def cone_wash(api, cal,
              n_rot: int     = CONE_N_ROT,
              amp_deg: float = CONE_AMP_DEG,
              speed: float | None = None) -> None:
    """Conical J5+J6 sweep at current (water dip) position."""
    if speed is None:
        speed = _speeds()["cone"]
    t_rot = 6.283 / speed
    print(f"    [cone wash]  {n_rot} rot × {amp_deg}°  speed={speed} rad/s  (~{t_rot*n_rot:.1f}s)")
    cone_sweep(api, np.array(cal["water_dip_q"]), n_rot=n_rot, amp_deg=amp_deg, speed=speed)


def lift_from_water(api, cal, speed: float | None = None) -> None:
    """After cone sweep: joint move back to water_hover_q (lift brush out of cup)."""
    if speed is None:
        speed = _speeds()["dip"]
    _joint_go(api, cal["water_hover_q"], speed, "↑ lift from water")


def drip_wait(secs: float | None = None) -> None:
    """Wait at Hover-2 for water to drip off brush."""
    if secs is None:
        secs = _speeds()["drip_sec"]
    print(f"    [drip wait]  {secs:.1f}s")
    time.sleep(secs)


# ── Compound sequences ────────────────────────────────────────────────────────

def wash_brush(api, cal,
               n_rot: int          = CONE_N_ROT,
               amp_deg: float      = CONE_AMP_DEG,
               wash_speed: float | None = None,
               drip_secs: float | None  = None) -> None:
    """Full wash cycle: water hover → dip → sweep → lift → drip."""
    goto_water_hover(api, cal)
    dip_water(api, cal)
    cone_wash(api, cal, n_rot=n_rot, amp_deg=amp_deg, speed=wash_speed)
    lift_from_water(api, cal)
    drip_wait(drip_secs)


def change_color(api, cal, new_slot: int,
                 n_rot: int          = CONE_N_ROT,
                 amp_deg: float      = CONE_AMP_DEG,
                 wash_speed: float | None = None,
                 drip_secs: float | None  = None) -> None:
    """Wash brush then dip into new_slot."""
    print(f"\n  == 换色 → {_slot_name(new_slot)} ==")
    wash_brush(api, cal, n_rot=n_rot, amp_deg=amp_deg,
               wash_speed=wash_speed, drip_secs=drip_secs)
    goto_paint_hover(api, cal, new_slot)
    dip_paint(api, cal, new_slot)
    goto_water_hover(api, cal)   # return to transit height


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slot_name(slot: int) -> str:
    from palette_cfg import SLOT_NAMES
    return SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
