"""
test4_paint.py — 实战画画：执行完整动作序列

读取:
  - data/trajectories/Tiger_actions.npz    动作序列 (paint/dip/wash)
  - data/calibration/palette.npy           调色盘 + 水筒位置
  - data/calibration/canvas.npy            画布标定 (来自 Cobrush Pro)

对每个动作:
  PAINT → pixel 坐标 → 画布物理坐标 → Cartesian 直线运动
  DIP   → 移动到对应颜料格蘸墨
  WASH  → 移动到水筒执行圆锥涮笔

Usage:
    python test4_paint.py
    python test4_paint.py --npz data/trajectories/Tiger_actions.npz
    python test4_paint.py --dry_run   (只打印不执行)
"""

import argparse
import sys
import time
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from palette_cfg   import PALETTE_NAMES, DEFAULT_CAL_PATH
from wash_action   import cone_trajectory, do_wash, CONE_SPEED, DIP_SPEED, HOVER_SPEED
from test3_dip_wash import dip_slot, wash   # reuse dip + wash logic
from config_loader import robot_ip
from pyfranka.franka_pybind import CartesianVelocities, CartesianVelocitiesFinished


ACTION_PAINT = 0
ACTION_DIP   = 1
ACTION_WASH  = 2

CANVAS_CAL_PATH = "data/calibration/canvas.npy"
DEFAULT_NPZ     = "data/trajectories/Tiger_actions.npz"

# Painting motion parameters
PAINT_HOVER_Z_OFFSET = 0.012   # metres above canvas surface for transit between strokes
PAINT_SPEED          = 0.15    # speed during stroke contact
TRANSIT_SPEED        = 0.6     # speed when moving between stroke start points


# ── Canvas transform ──────────────────────────────────────────────────────────

def load_canvas(path: str) -> dict:
    c = np.load(path, allow_pickle=True).item()
    return {
        "origin":   np.array(c["origin"]),
        "xyz_rot":  np.array(c["xyz_rot"]),
        "width_m":  float(c["width_m"]),
        "height_m": float(c["height_m"]),
        "z_canvas": float(c["z_canvas"]),
    }


def pixel_to_robot(px: float, py: float, canvas_px: int,
                   canvas: dict) -> np.ndarray:
    """Convert pixel coordinate to robot XYZ on the canvas plane."""
    u = px / canvas_px   # [0,1]
    v = py / canvas_px
    uv_m = np.array([u * canvas["width_m"], v * canvas["height_m"], 0.0])
    return canvas["origin"] + canvas["xyz_rot"] @ uv_m


def stroke_to_se3_pair(curve: np.ndarray, canvas_px: int,
                        canvas: dict, z_offset: float = 0.0):
    """Convert a 2-point pixel curve to (T_start, T_end) SE3 matrices."""
    z = canvas["z_canvas"] - z_offset   # negative = push into canvas slightly
    rot = canvas["xyz_rot"]             # columns are canvas X, Y, Z axes

    # Build EE orientation: Z-axis points into canvas, X along stroke direction
    dx = curve[1, 0] - curve[0, 0]
    dy = curve[1, 1] - curve[0, 1]
    stroke_dir = np.array([dx, dy, 0.0])
    norm = np.linalg.norm(stroke_dir)
    if norm > 1e-6:
        stroke_dir /= norm

    # EE orientation: x=stroke, z=-canvas_normal (pointing down to canvas)
    z_ee   = -rot[:, 2]                         # into canvas
    x_ee   = rot @ np.array([stroke_dir[0], stroke_dir[1], 0.0])
    x_ee  -= np.dot(x_ee, z_ee) * z_ee
    x_ee  /= max(np.linalg.norm(x_ee), 1e-9)
    y_ee   = np.cross(z_ee, x_ee)

    R = np.column_stack([x_ee, y_ee, z_ee])     # 3×3 rotation

    def make_T(xyz):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = xyz
        return T

    xyz0 = pixel_to_robot(curve[0, 0], curve[0, 1], canvas_px, canvas)
    xyz1 = pixel_to_robot(curve[1, 0], curve[1, 1], canvas_px, canvas)
    xyz0[2] = z;  xyz1[2] = z

    return make_T(xyz0), make_T(xyz1)


# ── Execution ─────────────────────────────────────────────────────────────────

def execute(api, npz_path: str, palette_cal: dict, canvas: dict,
            dry_run: bool = False) -> None:
    data = np.load(npz_path, allow_pickle=True)
    n_actions   = int(data["n_actions"])
    action_types = data["action_types"]
    curves      = data["curves"]
    colors      = data["colors"]
    widths      = data["widths"]
    slots       = data["slots"]
    canvas_px   = int(data["canvas_width"])

    ref_hover_T = np.array(palette_cal["ref_hover_T"])
    z_canvas    = canvas["z_canvas"]
    current_slot = -1

    n_paint = sum(1 for t in action_types if int(t) == ACTION_PAINT)
    n_dip   = sum(1 for t in action_types if int(t) == ACTION_DIP)
    n_wash  = sum(1 for t in action_types if int(t) == ACTION_WASH)
    print(f"\n  动作序列: {n_actions} 个  ({n_paint} paint / {n_dip} dip / {n_wash} wash)")
    if dry_run:
        print("  [DRY RUN — 只打印不执行]\n")

    paint_count = 0

    for i in range(n_actions):
        atype = int(action_types[i])
        slot  = int(slots[i])

        # ── WASH ─────────────────────────────────────────────────────────────
        if atype == ACTION_WASH:
            print(f"\n  [{i+1}/{n_actions}] WASH")
            if not dry_run:
                wash(api, palette_cal, n_rot=2, amp_deg=5)

        # ── DIP ──────────────────────────────────────────────────────────────
        elif atype == ACTION_DIP:
            name = PALETTE_NAMES[slot] if slot >= 0 else "?"
            print(f"\n  [{i+1}/{n_actions}] DIP  slot {slot} ({name})")
            if not dry_run:
                dip_slot(api, palette_cal, slot, ref_hover_T)
            current_slot = slot

        # ── PAINT ────────────────────────────────────────────────────────────
        elif atype == ACTION_PAINT:
            paint_count += 1
            curve = curves[i]
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])

            T_start, T_end = stroke_to_se3_pair(curve, canvas_px, canvas)

            # Hover start (above canvas)
            T_hover = T_start.copy()
            T_hover[2, 3] += PAINT_HOVER_Z_OFFSET

            if paint_count % 10 == 1:
                print(f"  [{i+1}/{n_actions}] PAINT {paint_count}/{n_paint}"
                      f"  slot={current_slot}  RGB=({r},{g},{b})")

            if not dry_run:
                for T_tgt, spd in [
                    (T_hover, TRANSIT_SPEED),
                    (T_start, DIP_SPEED),
                    (T_end,   PAINT_SPEED),
                    (T_hover, DIP_SPEED),
                ]:
                    p_goal = np.array(T_tgt[:3, 3])
                    def _cb(rs, period, _p=p_goal, _s=spd):
                        T_c = np.array(rs.O_T_EE).reshape(4, 4, order='F')
                        err = _p - T_c[:3, 3]
                        d = np.linalg.norm(err)
                        if d < 0.001:
                            return CartesianVelocitiesFinished(
                                CartesianVelocities([0, 0, 0, 0, 0, 0]))
                        v = (err / d) * min(_s, d * 3.0)
                        return CartesianVelocities(v.tolist() + [0, 0, 0])
                    api.robot_control(cartesian_velocities_handle=_cb)

    print(f"\n  ✓ 完成  {paint_count} 笔\n")


def main():
    p = argparse.ArgumentParser(description="实战画画: 执行完整 paint/dip/wash 序列")
    p.add_argument("--npz",      default=DEFAULT_NPZ,   help="动作序列 NPZ")
    p.add_argument("--cal",      default=DEFAULT_CAL_PATH, help="调色盘标定文件")
    p.add_argument("--canvas",   default=CANVAS_CAL_PATH,  help="画布标定文件")
    p.add_argument("--dry_run",  action="store_true",   help="只打印序列，不执行")
    args = p.parse_args()

    # ── Load calibrations ─────────────────────────────────────────────────────
    for fpath in [args.npz, args.cal, args.canvas]:
        if not Path(fpath).exists():
            print(f"[ERROR] 找不到文件: {fpath}")
            sys.exit(1)

    palette_cal = np.load(args.cal, allow_pickle=True).item()
    canvas      = load_canvas(args.canvas)

    print(f"\n  画布: {canvas['width_m']*100:.1f} cm × {canvas['height_m']*100:.1f} cm"
          f"  z={canvas['z_canvas']:.4f} m")

    if args.dry_run:
        execute(None, args.npz, palette_cal, canvas, dry_run=True)
        return

    try:
        from pyfranka.franka_pybind import FrankaApi
    except ImportError:
        print("[ERROR] pyfranka 未找到")
        sys.exit(1)

    ip = robot_ip()
    print(f"\n  连接机械臂 {ip} …")
    api = FrankaApi()
    api.init_config(ip, log_size=1000)
    api.set_default_behavior()
    st = api.readOnce()
    if st.robot_mode.name == "kReflex":
        api.automatic_error_recovery()
    print("  就绪。\n")

    input("  ⚠  确认颜料已配好、画布已固定、水筒已就位，按 Enter 开始 … ")
    execute(api, args.npz, palette_cal, canvas)


if __name__ == "__main__":
    main()
