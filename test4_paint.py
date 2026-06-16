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
from franka import Franka, CartesianVelocities, CartesianVelocitiesFinished
from config_loader import robot_ip, load_config
from palette_cfg   import SLOT_NAMES, DEFAULT_CAL_PATH, ACTION_PAINT, ACTION_DIP, ACTION_WASH
from palette_actions import (
    go_home, goto_water_hover,
    goto_paint_hover, dip_paint,
    wash_brush, _speeds,
)

CANVAS_CAL_PATH = "data/calibration/canvas.npy"
DEFAULT_NPZ     = "data/trajectories/Tiger_actions.npz"

PAINT_SPEED   = 0.048  # m/s during stroke contact (−20% from 0.06)
TRANSIT_SPEED = 0.12   # m/s hover transit between strokes (−20% from 0.15)
HOVER_LIFT    = 0.015  # m above canvas z for hover between strokes
TAU_SMOOTH    = 0.10   # velocity smoothing time constant (s)


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


def stroke_xyz(curve: np.ndarray, canvas: dict,
               z_press: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert 2-point pixel curve to (xyz_start, xyz_end) on canvas."""
    z = canvas["z_canvas"] + z_press
    p0 = pixel_to_robot(curve[0, 0], curve[0, 1], canvas);  p0[2] = z
    p1 = pixel_to_robot(curve[1, 0], curve[1, 1], canvas);  p1[2] = z
    return p0, p1


# ── Smooth Cartesian controller ───────────────────────────────────────────────

def _cart_go(robot: Franka, target_xyz, speed: float) -> None:
    """Smooth P-controller Cartesian move (exponential velocity filter)."""
    p_goal = np.array(target_xyz, dtype=float)
    v_cur  = np.zeros(3)

    def cb(rs, period):
        dt  = max(period.toSec(), 0.0005)
        T_c = np.array(rs.O_T_EE).reshape(4, 4, order='F')
        err = p_goal - T_c[:3, 3]
        d   = np.linalg.norm(err)
        if d < 0.001 and np.linalg.norm(v_cur) < 0.005:
            return CartesianVelocitiesFinished(CartesianVelocities([0.0] * 6))
        v_des     = (err / d) * min(speed, d * 4.0) if d > 0.001 else np.zeros(3)
        alpha     = 1.0 - math.exp(-dt / TAU_SMOOTH)
        v_cur[:] += alpha * (v_des - v_cur)
        return CartesianVelocities(v_cur.tolist() + [0.0, 0.0, 0.0])

    robot.robot_control(cartesian_velocities_handle=cb)


# ── Execution ─────────────────────────────────────────────────────────────────

def execute(robot, npz_path: str, palette_cal: dict, canvas: dict,
            z_press: float = 0.0, dry_run: bool = False) -> None:
    data         = np.load(npz_path, allow_pickle=True)
    n_actions    = int(data["n_actions"])
    action_types = data["action_types"]
    curves       = data["curves"]
    slots        = data["slots"]

    n_paint = int(np.sum(action_types == ACTION_PAINT))
    n_dip   = int(np.sum(action_types == ACTION_DIP))
    n_wash  = int(np.sum(action_types == ACTION_WASH))
    print(f"\n  序列: {n_actions} 动作  ({n_paint} paint / {n_dip} dip / {n_wash} wash)")
    if dry_run:
        print("  [DRY RUN]\n")

    paint_count   = 0
    current_slot  = -1
    at_hover      = False   # track whether we're already at canvas hover height

    for i in range(n_actions):
        atype = int(action_types[i])
        slot  = int(slots[i])

        # ── WASH ─────────────────────────────────────────────────────────────
        if atype == ACTION_WASH:
            slot_name = SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
            print(f"\n  [{i+1}/{n_actions}] WASH  (before {slot_name})")
            if not dry_run:
                go_home(robot)
                wash_brush(robot, palette_cal)
            at_hover = False

        # ── DIP ──────────────────────────────────────────────────────────────
        elif atype == ACTION_DIP:
            slot_name = SLOT_NAMES[slot] if 0 <= slot < len(SLOT_NAMES) else f"slot{slot}"
            is_redip  = (slot == current_slot and current_slot != -1)
            tag       = "RE-DIP" if is_redip else "DIP"
            print(f"\n  [{i+1}/{n_actions}] {tag}  → {slot_name} (slot {slot})")
            if not dry_run:
                if not is_redip:
                    go_home(robot)
                goto_paint_hover(robot, palette_cal, slot)
                dip_paint(robot, palette_cal, slot)
                go_home(robot)
            current_slot = slot
            at_hover = False

        # ── PAINT ─────────────────────────────────────────────────────────────
        elif atype == ACTION_PAINT:
            paint_count += 1
            curve = curves[i]
            p0, p1 = stroke_xyz(curve, canvas, z_press)
            p0_hov = p0.copy();  p0_hov[2] += HOVER_LIFT
            p1_hov = p1.copy();  p1_hov[2] += HOVER_LIFT

            if paint_count % 20 == 1:
                print(f"  [{i+1}/{n_actions}] PAINT {paint_count}/{n_paint}"
                      f"  slot={current_slot}")

            if dry_run:
                continue

            # Transit to hover above stroke start
            _cart_go(robot, p0_hov, TRANSIT_SPEED)
            # Descend to canvas
            _cart_go(robot, p0, PAINT_SPEED)
            # Draw stroke
            _cart_go(robot, p1, PAINT_SPEED)
            # Lift off
            _cart_go(robot, p1_hov, PAINT_SPEED)
            at_hover = True

    if not dry_run:
        go_home(robot)
    print(f"\n  ✓ 完成  {paint_count} 笔\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="实战画画: paint/dip/wash 序列")
    p.add_argument("--npz",     default=DEFAULT_NPZ)
    p.add_argument("--cal",     default=DEFAULT_CAL_PATH)
    p.add_argument("--canvas",  default=CANVAS_CAL_PATH)
    p.add_argument("--z_press", type=float, default=0.0,
                   help="Z 偏移 (m)，负值=多按入画布，默认 0")
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
                z_press=args.z_press, dry_run=True)
        return

    ip    = robot_ip()
    robot = Franka(ip)
    if not robot.wait_ready():
        print("[ABORT] robot not ready")
        sys.exit(1)

    input("\n  ⚠  确认颜料已配好、画布已固定、水筒已就位，按 Enter 开始 … ")
    execute(robot, args.npz, palette_cal, canvas,
            z_press=args.z_press, dry_run=False)


if __name__ == "__main__":
    main()
