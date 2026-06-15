"""
test2_calibrate.py — 标定第一个颜色 + 水筒，计算并显示所有格子位置

只需手动引导4步:
  1. HOVER 第一个颜料格上方
  2. DIP   进颜料（配置蘸墨深度）
  3. HOVER 水筒上方
  4. DIP   进水筒（笔尖入水）

其他5个格子的坐标由栅格间距自动计算。

输出: data/calibration/palette.npy

Usage:
    python test2_calibrate.py --ip 192.170.10.200
    python test2_calibrate.py --ip 192.170.10.200 --ref_slot 0 --pitch_x 0.035 --pitch_y 0.035
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from palette_cfg import (
    PALETTE_RGB, PALETTE_NAMES, SLOT_GRID, N_SLOTS,
    SLOT_PITCH_X, SLOT_PITCH_Y, DEFAULT_CAL_PATH,
)


def _swatch(r, g, b):
    return f"\033[48;2;{r};{g};{b}m   \033[0m"


def _fmt(xyz):
    return f"[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]"


def main():
    p = argparse.ArgumentParser(description="标定调色盘: 第一个格子 + 水筒")
    p.add_argument("--ip",       required=True)
    p.add_argument("--ref_slot", type=int, default=0, choices=range(N_SLOTS),
                   help="参考格子索引 (默认 0 = Red)")
    p.add_argument("--pitch_x",  type=float, default=SLOT_PITCH_X,
                   help=f"列间距 m (默认 {SLOT_PITCH_X})")
    p.add_argument("--pitch_y",  type=float, default=SLOT_PITCH_Y,
                   help=f"行间距 m (默认 {SLOT_PITCH_Y})")
    p.add_argument("--out",      default=DEFAULT_CAL_PATH)
    args = p.parse_args()

    try:
        from pyfranka.franka_pybind import FrankaApi
    except ImportError:
        print("[ERROR] pyfranka 未找到")
        sys.exit(1)

    print(f"\n  连接机械臂 {args.ip} …")
    api = FrankaApi()
    api.init_config(args.ip, log_size=1000)
    api.set_default_behavior()
    st = api.readOnce()
    if st.robot_mode.name == "kReflex":
        api.automatic_error_recovery()
    print("  机械臂就绪。可手动引导（按住引导按钮）。\n")

    ref = args.ref_slot
    r, g, b = PALETTE_RGB[ref]
    name    = PALETTE_NAMES[ref]

    def record(step_label):
        st = api.readOnce()
        T  = np.array(st.O_T_EE).reshape(4, 4, order='F')
        xyz = T[:3, 3].copy()
        q   = list(st.q)
        print(f"       当前 EE: {_fmt(xyz)}")
        input(f"       {step_label}  [Enter 记录] ")
        st  = api.readOnce()
        T   = np.array(st.O_T_EE).reshape(4, 4, order='F')
        xyz = T[:3, 3].copy()
        q   = list(st.q)
        T_full = T.copy()
        print(f"       已记录: {_fmt(xyz)}\n")
        return xyz, q, T_full

    # ── Step 1: Hover over ref slot ───────────────────────────────────────────
    print(f"  ── 步骤 1/4  HOVER — 参考格子 {ref} {_swatch(r,g,b)} {name} 上方 ──")
    hover_xyz, hover_q, hover_T = record("引导笔到颜料格上方 (安全高度)")

    # ── Step 2: Dip into ref slot ─────────────────────────────────────────────
    print(f"  ── 步骤 2/4  DIP — 蘸墨到 {name} ──")
    dip_xyz, dip_q, dip_T = record("引导笔尖进入颜料 (合适深度 = 蘸墨深度标定)")

    hover_z_offset = float(hover_xyz[2] - dip_xyz[2])
    print(f"  蘸墨深度 (hover Z − dip Z): {hover_z_offset*1000:.1f} mm\n")

    # ── Step 3: Hover over water cup ──────────────────────────────────────────
    print("  ── 步骤 3/4  HOVER — 水筒上方 ──")
    water_hover_xyz, water_hover_q, _ = record("引导笔到水筒上方")

    # ── Step 4: Dip into water cup ────────────────────────────────────────────
    print("  ── 步骤 4/4  DIP — 笔尖入水 (中心位置) ──")
    water_dip_xyz, water_dip_q, _ = record("引导笔尖进水 (圆锥扫掠的定点)")

    # ── Compute all slot positions ────────────────────────────────────────────
    ref_row, ref_col = SLOT_GRID[ref]

    print(f"\n  ═══ 所有格子计算坐标 (pitch x={args.pitch_x*1000:.1f}mm y={args.pitch_y*1000:.1f}mm) ═══")
    print(f"  {'槽':>4}  Swatch  {'名称':<10}  {'DIP XYZ (m)':<42}  {'HOVER Z (m)'}")
    print(f"  {'─'*80}")

    slot_xyz_all = {}
    for i in range(N_SLOTS):
        ri, ci     = SLOT_GRID[i]
        dr, dc     = ri - ref_row, ci - ref_col
        dip_i_xyz  = [
            dip_xyz[0] + dc * args.pitch_x,
            dip_xyz[1] + dr * args.pitch_y,
            dip_xyz[2],
        ]
        hover_i_xyz = list(dip_i_xyz)
        hover_i_xyz[2] += hover_z_offset
        slot_xyz_all[i] = {"dip": dip_i_xyz, "hover": hover_i_xyz}

        ri_c, gi_c, bi_c = PALETTE_RGB[i]
        tag = " ← ref" if i == ref else ""
        print(f"  {i:4d}   {_swatch(ri_c,gi_c,bi_c)}  {PALETTE_NAMES[i]:<10}  "
              f"{_fmt(dip_i_xyz):<42}  {hover_i_xyz[2]:.4f}{tag}")

    # ── Save ──────────────────────────────────────────────────────────────────
    cal = {
        "ref_slot":        ref,
        "ref_hover_xyz":   hover_xyz.tolist(),
        "ref_dip_xyz":     dip_xyz.tolist(),
        "ref_hover_T":     hover_T.tolist(),   # full SE3 for computing other slots
        "ref_hover_q":     hover_q,
        "ref_dip_q":       dip_q,
        "hover_z_offset":  hover_z_offset,
        "slot_pitch_xy":   [args.pitch_x, args.pitch_y],
        "water_hover_xyz": water_hover_xyz.tolist(),
        "water_cup_xyz":   water_dip_xyz.tolist(),
        "water_hover_q":   water_hover_q,
        "water_dip_q":     water_dip_q,
        "slot_xyz_all":    slot_xyz_all,       # pre-computed for quick lookup
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), cal)
    print(f"\n  ✓ 保存 → {out}\n")


if __name__ == "__main__":
    main()
