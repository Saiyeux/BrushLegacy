"""
test4_paint.py — 完整画画流程

读取:
  data/trajectories/<name>_actions.npz   动作序列 (PAINT / DIP / WASH)
  data/calibration/palette.npy           调色盘 + 水筒位置
  data/calibration/canvas.npy            画布标定

NPZ 格式 (MacBook 侧生成):
  n_actions    : int
  action_types : (N,)  int  —  0=PAINT  1=DIP  2=WASH
  curves       : (N, 2, 2)  float  —  像素坐标 [[x0,y0],[x1,y1]]
  slots        : (N,)  int  —  DIP/WASH 时对应颜料槽
  canvas_width : int   —  生成时的图像边长（用于归一化）

流程:
  go_home → [DIP slot] → go_home → [stroke, stroke, …] → (repeat) → go_home

Usage:
    python test4_paint.py --npz data/trajectories/Tiger_actions.npz
    python test4_paint.py --npz data/trajectories/Tiger_actions.npz --dry_run
    python test4_paint.py --npz data/trajectories/Tiger_actions.npz --z_press -0.003
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "src/robot")
from franka import Franka, J7_PIN, CartesianVelocities, CartesianVelocitiesFinished
from config_loader import robot_ip, load_config
from palette_cfg   import SLOT_NAMES, DEFAULT_CAL_PATH, ACTION_PAINT, ACTION_DIP, ACTION_WASH, slot_xyz
from palette_actions import (
    go_home, goto_water_hover,
    goto_paint_hover, dip_paint,
    wash_brush, _speeds,
)
from log import tlog, tlog_reset

CANVAS_CAL_PATH = "data/calibration/canvas.npy"
DEFAULT_NPZ     = "data/trajectories/Tiger_actions.npz"

PAINT_SPEED    = 0.048  # m/s during stroke contact
TRANSIT_SPEED  = 0.12   # m/s hover transit between strokes
HOVER_LIFT     = 0.015  # m above canvas z for hover between strokes
INTER_OP_LIFT  = 0.05   # m above the highest hover (canvas or palette) for safe transits
TAU_SMOOTH     = 0.10   # velocity smoothing time constant (s)

# Canvas-tilt compensation: the table/canvas is not perfectly horizontal.
# Z rises linearly across the canvas surface.
# CANVAS_TILT_X: total Z difference from left edge to right edge (metres)
# CANVAS_TILT_Y: total Z difference from top edge to bottom edge (metres)
CANVAS_TILT_X = 0.0018   # m  — 1.8 mm height difference across canvas width
CANVAS_TILT_Y = 0.0026   # m  — 2.6 mm height difference across canvas height


# ── Canvas helpers ────────────────────────────────────────────────────────────

def load_canvas(path: str) -> dict:
    c = np.load(path, allow_pickle=True).item()
    return {
        "origin":   np.array(c["origin"]),
        "xyz_rot":  np.array(c["xyz_rot"]),
        "width_m":  float(c["width_m"]),
        "height_m": float(c["height_m"]),
        "z_canvas": float(c["z_canvas"]),
        "width_px": int(c.get("width_px", 512)),
    }


def pixel_to_robot(px: float, py: float, canvas: dict) -> np.ndarray:
    """Pixel coordinate (image space) → robot XYZ on canvas plane."""
    u = px / canvas["width_px"]
    v = py / canvas["width_px"]
    uv_m = np.array([u * canvas["width_m"], v * canvas["height_m"], 0.0])
    return canvas["origin"] + canvas["xyz_rot"] @ uv_m


def _tilt_z(px: float, py: float, canvas: dict) -> float:
    """Z correction for canvas tilt at pixel position (px, py).

    The canvas plane is not horizontal: Z varies linearly from edge to edge.
    u, v ∈ [0, 1] are normalised canvas coordinates (same normalisation as
    pixel_to_robot). The correction is added on top of z_canvas.
    """
    u = px / canvas["width_px"]
    v = py / canvas["width_px"]
    return u * CANVAS_TILT_X + v * CANVAS_TILT_Y


def stroke_xyz(curve: np.ndarray, canvas: dict,
               z_press: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert 2-point pixel curve to (xyz_start, xyz_end) on canvas.

    Z is adjusted per-point based on canvas tilt so the brush maintains
    consistent contact pressure across a non-horizontal table.
    """
    def _pt(px, py):
        p    = pixel_to_robot(px, py, canvas)
        p[2] = canvas["z_canvas"] + z_press + _tilt_z(px, py, canvas)
        return p

    return _pt(curve[0, 0], curve[0, 1]), _pt(curve[1, 0], curve[1, 1])


# ── Smooth Cartesian controller ───────────────────────────────────────────────

K_ORI = 1.5   # rad/s per rad of orientation error
W_MAX = 0.25  # max angular velocity command (rad/s)


def _cart_go(robot: Franka, target_xyz, speed: float,
             q7_target: float | None = None,
             R_target: np.ndarray | None = None) -> None:
    """Smooth P-controller Cartesian move with optional orientation hold.

    R_target: 3×3 rotation matrix for the desired end-effector orientation.
    When set, angular velocity feedback is added to resist flange tilt.
    q7_target: J7 null-space pin (via robot_control_j7_pinned).
    """
    p_goal = np.array(target_xyz, dtype=float)
    v_cur  = np.zeros(3)

    def cb(rs, period):
        dt  = max(period.toSec(), 0.0005)
        T_c = np.array(rs.O_T_EE).reshape(4, 4, order='F')

        # ── Position ──────────────────────────────────────────────────────────
        err = p_goal - T_c[:3, 3]
        d   = np.linalg.norm(err)
        if d < 0.001 and np.linalg.norm(v_cur) < 0.005:
            return CartesianVelocitiesFinished(CartesianVelocities([0.0] * 6))
        v_des     = (err / d) * min(speed, d * 4.0) if d > 0.001 else np.zeros(3)
        alpha     = 1.0 - math.exp(-dt / TAU_SMOOTH)
        v_cur[:] += alpha * (v_des - v_cur)

        # ── Orientation ───────────────────────────────────────────────────────
        if R_target is not None:
            R_c   = T_c[:3, :3]
            R_err = R_target @ R_c.T          # rotation from current → target
            tr    = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            angle = math.acos(tr)
            if angle > 1e-6:
                s  = 2.0 * math.sin(angle)
                ax = np.array([R_err[2, 1] - R_err[1, 2],
                               R_err[0, 2] - R_err[2, 0],
                               R_err[1, 0] - R_err[0, 1]]) / s
                w  = np.clip(K_ORI * angle * ax, -W_MAX, W_MAX)
            else:
                w = np.zeros(3)
        else:
            w = np.zeros(3)

        return CartesianVelocities(v_cur.tolist() + w.tolist())

    if q7_target is not None:
        robot.api.robot_control_j7_pinned(cb, q7_target)
    else:
        robot.robot_control(cartesian_velocities_handle=cb)


# ── Execution ─────────────────────────────────────────────────────────────────

def _scan_start(action_types, slots, start_stroke: int) -> tuple[int, int]:
    """Return (action_idx, slot) where the start_stroke-th PAINT occurs.

    slot is the colour that should be on the brush at that point.
    """
    paint_count  = 0
    current_slot = -1
    for i, atype in enumerate(action_types):
        if int(atype) == ACTION_DIP:
            current_slot = int(slots[i])
        if int(atype) == ACTION_PAINT:
            paint_count += 1
            if paint_count >= start_stroke:
                return i, current_slot
    return len(action_types), current_slot


def execute(robot, npz_path: str, palette_cal: dict, canvas: dict,
            z_press: float = 0.0, dry_run: bool = False,
            start_stroke: int = 1) -> None:
    data         = np.load(npz_path, allow_pickle=True)
    n_actions    = int(data["n_actions"])
    action_types = data["action_types"]
    curves       = data["curves"]
    slots        = data["slots"]

    q7 = J7_PIN   # held throughout entire painting flow

    n_paint = int(np.sum(action_types == ACTION_PAINT))
    n_dip   = int(np.sum(action_types == ACTION_DIP))
    n_wash  = int(np.sum(action_types == ACTION_WASH))

    tlog_reset()
    tlog(f"序列: {n_actions} 动作  ({n_paint} paint / {n_dip} dip / {n_wash} wash)"
         + ("  [DRY RUN]" if dry_run else ""))

    # ── Capture target flange orientation + compute safe transit Z ────────────
    R_paint:    np.ndarray | None = None
    inter_op_z: float             = 0.30   # fallback; recomputed below

    if not dry_run:
        go_home(robot)
        st      = robot.read_state()
        T_home  = np.array(st.O_T_EE).reshape(4, 4, order='F')
        R_paint = T_home[:3, :3].copy()
        tlog("姿态基准捕获完毕")

    canvas_hover_z = canvas["z_canvas"] + HOVER_LIFT
    slot_hover_zs  = [slot_xyz(palette_cal, s, "hover")[2]
                      for s in range(len(palette_cal.get("slot_hover_xyz", [])))]
    inter_op_z = max([canvas_hover_z] + slot_hover_zs) + INTER_OP_LIFT

    # ── Start-stroke fast-forward ─────────────────────────────────────────────
    if start_stroke > 1:
        start_idx, pre_slot = _scan_start(action_types, slots, start_stroke)
        slot_name = SLOT_NAMES[pre_slot] if 0 <= pre_slot < len(SLOT_NAMES) else f"slot{pre_slot}"
        tlog(f"↷ 跳到第 {start_stroke} 笔  (从 action {start_idx}, 颜色={slot_name})")
        if not dry_run and pre_slot >= 0:
            goto_paint_hover(robot, palette_cal, pre_slot)
            dip_paint(robot, palette_cal, pre_slot)
            # Rise to inter_op_z: ready for first stroke
            hov = slot_xyz(palette_cal, pre_slot, "hover")
            _cart_go(robot, [hov[0], hov[1], inter_op_z], TRANSIT_SPEED, q7, R_paint)
        action_range = range(start_idx, n_actions)
        paint_count  = start_stroke - 1
        current_slot = pre_slot
        at_op_z      = not dry_run   # after DIP we're at inter_op_z
    else:
        action_range = range(n_actions)
        paint_count  = 0
        current_slot = -1
        at_op_z      = False   # starts at home (joint space)

    for i in action_range:
        atype = int(action_types[i])
        slot  = int(slots[i])

        # ── WASH ─────────────────────────────────────────────────────────────
        if atype == ACTION_WASH:
            slot_name = SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
            tlog(f"WASH  [{i+1}/{n_actions}]  before {slot_name}")
            if not dry_run:
                # go_home is safe from inter_op_z; canvas is below us
                go_home(robot)
                wash_brush(robot, palette_cal)
            at_op_z = False   # after wash: at water_hover (joint space)

        # ── DIP ──────────────────────────────────────────────────────────────
        elif atype == ACTION_DIP:
            slot_name = SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
            is_redip  = (slot == current_slot and current_slot != -1)
            tag       = "RE-DIP" if is_redip else "DIP"
            tlog(f"{tag}  [{i+1}/{n_actions}]  → {slot_name}")

            if not dry_run:
                if at_op_z:
                    # Safe Cartesian path: already at inter_op_z, transit to above slot
                    hov = slot_xyz(palette_cal, slot, "hover")
                    _cart_go(robot, [hov[0], hov[1], inter_op_z],
                             TRANSIT_SPEED, q7, R_paint)
                    # Joint-space to slot_hover_q for correct flange orientation
                    goto_paint_hover(robot, palette_cal, slot)
                else:
                    # Coming from home or water area — joint-space go_home then palette
                    go_home(robot)
                    goto_paint_hover(robot, palette_cal, slot)

                dip_paint(robot, palette_cal, slot)

                # Rise back to inter_op_z after dip (Cartesian, no canvas risk)
                hov = slot_xyz(palette_cal, slot, "hover")
                _cart_go(robot, [hov[0], hov[1], inter_op_z],
                         TRANSIT_SPEED, q7, R_paint)

            current_slot = slot
            at_op_z      = not dry_run

        # ── PAINT ─────────────────────────────────────────────────────────────
        elif atype == ACTION_PAINT:
            paint_count += 1
            curve = curves[i]
            p0, p1 = stroke_xyz(curve, canvas, z_press)
            p0_hov = p0.copy();  p0_hov[2] += HOVER_LIFT
            p1_hov = p1.copy();  p1_hov[2] += HOVER_LIFT

            tlog(f"PAINT {paint_count}/{n_paint}  [{i+1}/{n_actions}]")

            if dry_run:
                continue

            if at_op_z:
                # At inter_op_z: Cartesian transit to above stroke start, then descend
                _cart_go(robot, [p0[0], p0[1], inter_op_z],
                         TRANSIT_SPEED, q7, R_paint)
            _cart_go(robot, p0_hov, TRANSIT_SPEED, q7, R_paint)
            _cart_go(robot, p0,     PAINT_SPEED,   q7, R_paint)
            _cart_go(robot, p1,     PAINT_SPEED,   q7, R_paint)
            _cart_go(robot, p1_hov, PAINT_SPEED,   q7, R_paint)
            # Rise to inter_op_z immediately after stroke — never go_home mid-painting
            _cart_go(robot, [p1[0], p1[1], inter_op_z],
                     TRANSIT_SPEED, q7, R_paint)
            at_op_z = True

    if not dry_run:
        go_home(robot)
    tlog(f"✓ 完成  {paint_count} 笔")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="实战画画: paint/dip/wash 序列")
    p.add_argument("--npz",     default=DEFAULT_NPZ)
    p.add_argument("--cal",     default=DEFAULT_CAL_PATH)
    p.add_argument("--canvas",  default=CANVAS_CAL_PATH)
    p.add_argument("--z_press", type=float, default=0.0,
                   help="Z 偏移 (m)，负值=多按入画布，默认 0")
    p.add_argument("--start_stroke", type=int, default=1,
                   help="从第几笔开始执行（默认 1，即从头开始）")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    for fpath in [args.npz, args.cal, args.canvas]:
        if not Path(fpath).exists():
            print(f"[ERROR] 找不到文件: {fpath}")
            sys.exit(1)

    palette_cal = np.load(args.cal,    allow_pickle=True).item()
    canvas      = load_canvas(args.canvas)

    print(f"\n  画布: {canvas['width_m']*100:.1f} × {canvas['height_m']*100:.1f} cm"
          f"  z={canvas['z_canvas']:.4f} m")
    if args.z_press:
        print(f"  z_press: {args.z_press*1000:+.1f} mm")

    if args.dry_run:
        execute(None, args.npz, palette_cal, canvas,
                z_press=args.z_press, dry_run=True,
                start_stroke=args.start_stroke)
        return

    ip    = robot_ip()
    robot = Franka(ip)
    if not robot.wait_ready():
        print("[ABORT] robot not ready")
        sys.exit(1)

    input("\n  ⚠  确认颜料已配好、画布已固定、水筒已就位，按 Enter 开始 … ")
    execute(robot, args.npz, palette_cal, canvas,
            z_press=args.z_press, dry_run=False,
            start_stroke=args.start_stroke)


if __name__ == "__main__":
    main()
