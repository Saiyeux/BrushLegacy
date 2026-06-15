"""
test2_calibrate.py — 标定调色盘 + 水筒位置

4 个交互步骤:
  1. 拖动到 Red(0,0) Hover-1 位置（颜料格正上方）→ 记录关节角 + SE3
  2. 拖动到 Yellow(0,4) Hover-1 位置（同行第4格）→ 确定列方向
  3. 拖动到水筒 Hover-2 位置（水筒正上方，同时作为过渡高度）→ 记录关节角
  4. 拖动到水筒 Dip 位置（笔尖入水中心，圆锥扫掠定点）→ 记录关节角

由 Red + Yellow 自动计算列方向，由垂直关系推算行方向。
所有颜料格位置从 Red 参考点按格子间距计算。
蘸墨深度从 config.yaml 读取 palette.dip_depth_mm。

Usage:
    python test2_calibrate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from palette_cfg import (
    SLOT_NAMES, SLOT_RGB, SLOT_GRID, N_SLOTS,
    REF_SLOT, REF_SLOT2, DEFAULT_CAL_PATH,
)
from config_loader import load_config, robot_ip


def _swatch(r, g, b):
    return f"\033[48;2;{r};{g};{b}m   \033[0m"


def _fmt(xyz):
    return f"[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]"


def _record(robot, prompt: str):
    """Wait for user, then read current EE pose and joint angles."""
    input(f"\n  → {prompt}\n    就位后按 Enter 记录 … ")
    st  = robot.read_state()
    T   = np.array(st.O_T_EE).reshape(4, 4, order='F')
    q   = list(st.q)
    xyz = T[:3, 3].copy()
    print(f"    已记录: {_fmt(xyz)}")
    return xyz, q, T


def main():
    cfg          = load_config()
    cell_w_m     = cfg["palette"]["cell_width_mm"]  / 1000.0   # column pitch
    cell_h_m     = cfg["palette"]["cell_height_mm"] / 1000.0   # row pitch
    dip_depth    = cfg["palette"]["dip_depth_mm"]   / 1000.0

    ip = robot_ip()

    out_path = Path(DEFAULT_CAL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  调色盘标定")
    print(f"  格子宽: {cell_w_m*1000:.0f} mm  高: {cell_h_m*1000:.0f} mm   蘸墨深度: {dip_depth*1000:.0f} mm")
    print(f"{'='*60}")
    print("  操作方式: 按住引导按钮手动拖动机械臂，就位后松开，按 Enter\n")

    from franka import Franka
    robot = Franka(ip)
    if not robot.wait_ready():
        print("[ABORT] robot not ready")
        sys.exit(1)
    print("  就绪。\n")

    # ── Step 1: Red (0,0) Hover-1 ─────────────────────────────────────────────
    r0, g0, b0 = SLOT_RGB[REF_SLOT]
    print(f"  ─── 步骤 1/4  Red {_swatch(r0,g0,b0)} (0,0) Hover-1 ───")
    print("  Hover-1 = 颜料格正上方（蘸墨起始高度）")
    red_xyz, red_q, red_T = _record(robot, "拖动到 Red 颜料格正上方")

    # ── Step 2: Yellow (0,4) Hover-1 ─────────────────────────────────────────
    r1, g1, b1 = SLOT_RGB[REF_SLOT2]
    print(f"\n  ─── 步骤 2/4  Yellow {_swatch(r1,g1,b1)} (0,4) Hover-1 ───")
    print("  同一行，向右数第4格（用于确定列方向）")
    yellow_xyz, _, _ = _record(robot, "拖动到 Yellow 颜料格正上方")

    # Compute grid axes
    col_vec  = yellow_xyz - red_xyz                     # Red → Yellow: 4 cols
    col_dir  = col_vec / np.linalg.norm(col_vec)       # unit vector
    col_n_cells = SLOT_GRID[REF_SLOT2][1] - SLOT_GRID[REF_SLOT][1]  # = 4
    col_pitch_measured = np.linalg.norm(col_vec) / col_n_cells
    print(f"\n  列方向: {[f'{v:.3f}' for v in col_dir]}")
    print(f"  列实测间距: {col_pitch_measured*1000:.1f} mm  (config cell_width: {cell_w_m*1000:.0f} mm)")

    z_up    = np.array([0.0, 0.0, 1.0])
    row_dir = np.cross(z_up, col_dir)
    row_dir /= np.linalg.norm(row_dir)
    print(f"  行方向: {[f'{v:.3f}' for v in row_dir]}")
    print(f"  行间距 (config cell_height): {cell_h_m*1000:.0f} mm")

    # ── Step 3: Water Hover-2 ─────────────────────────────────────────────────
    print(f"\n  ─── 步骤 3/4  水筒 Hover-2（过渡高度）───")
    print("  Hover-2 = 水筒正上方，也是所有过渡动作的通道高度")
    water_hover_xyz, water_hover_q, _ = _record(robot, "拖动到水筒正上方")

    # ── Step 4: Water Dip ─────────────────────────────────────────────────────
    print(f"\n  ─── 步骤 4/4  水筒 Dip（圆锥扫掠定点）───")
    print("  笔尖入水中心，此处将执行 J5+J6 圆锥扫掠")
    water_dip_xyz, water_dip_q, _ = _record(robot, "拖动到笔尖入水位置")

    # ── Compute all slot positions ─────────────────────────────────────────────
    ref_row, ref_col = SLOT_GRID[REF_SLOT]
    slot_hover_T = {}
    slot_dip_T   = {}

    print(f"\n  ═══ 所有格子计算位置 ═══")
    print(f"  {'槽':>3}  Swatch  {'名称':<8}  {'Grid':>6}  {'Hover-1 XYZ':^38}  Dip Z")
    print(f"  {'─'*80}")

    for i in range(N_SLOTS):
        ri, ci = SLOT_GRID[i]
        dr = ri - ref_row
        dc = ci - ref_col
        hover_xyz = red_xyz + dr * cell_h_m * row_dir + dc * cell_w_m * col_dir
        dip_xyz   = hover_xyz.copy()
        dip_xyz[2] -= dip_depth

        # Build SE3 (keep Red's orientation, change translation)
        T_hover = red_T.copy()
        T_hover[:3, 3] = hover_xyz
        T_dip   = red_T.copy()
        T_dip[:3, 3]   = dip_xyz

        slot_hover_T[i] = T_hover.tolist()
        slot_dip_T[i]   = T_dip.tolist()

        r, g, b = SLOT_RGB[i]
        tag = " ← ref" if i == REF_SLOT else (" ← ref2" if i == REF_SLOT2 else "")
        print(f"  {i:3d}   {_swatch(r,g,b)}  {SLOT_NAMES[i]:<8}  "
              f"({ri},{ci})  {_fmt(hover_xyz)}  {dip_xyz[2]:.4f}{tag}")

    # ── Save ──────────────────────────────────────────────────────────────────
    cal = {
        # Red reference
        "red_hover_q":   red_q,
        "red_hover_T":   red_T.tolist(),
        "red_hover_xyz": red_xyz.tolist(),
        # Grid axes
        "col_dir":       col_dir.tolist(),
        "row_dir":       row_dir.tolist(),
        "cell_w_m":      float(cell_w_m),
        "cell_h_m":      float(cell_h_m),
        "dip_depth_m":   float(dip_depth),
        # Water cup
        "water_hover_q":   water_hover_q,
        "water_hover_xyz": water_hover_xyz.tolist(),
        "water_dip_q":     water_dip_q,
        "water_dip_xyz":   water_dip_xyz.tolist(),
        # Pre-computed slot positions
        "slot_hover_T":  slot_hover_T,
        "slot_dip_T":    slot_dip_T,
    }

    np.save(str(out_path), cal)
    print(f"\n  ✓ 保存 → {out_path}\n")


if __name__ == "__main__":
    main()
