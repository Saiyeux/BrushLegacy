"""
test3_dip_wash.py — 轮流蘸色+涮笔，共6次

按标定结果，依次对 6 个颜料格执行:
  1. 移动到格子 hover 位置
  2. 下降蘸墨 (DIP)
  3. 回到 hover
  4. 移动到水筒 hover
  5. 涮笔 (J5+J6 圆锥扫掠)
  6. 回到水筒 hover

全程使用从 test2_calibrate.py 保存的 palette.npy。
其他格子位置由参考格子坐标 + 栅格间距计算，用 Cartesian 直线运动到达。

Usage:
    python test3_dip_wash.py --ip 192.170.10.200
    python test3_dip_wash.py --ip 192.170.10.200 --cal data/calibration/palette.npy --n 2 --amp 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from palette_cfg import PALETTE_RGB, PALETTE_NAMES, N_SLOTS, DEFAULT_CAL_PATH
from wash_action  import cone_trajectory, CONE_SPEED, DIP_SPEED, HOVER_SPEED


def _swatch(r, g, b):
    return f"\033[48;2;{r};{g};{b}m   \033[0m"


def _go_joint(api, q, speed, label=""):
    print(f"    → {label}")
    api.joint_go(q if isinstance(q, list) else q.tolist(), speed=speed)


def _go_cart(api, xyz, hover_T, hover_xyz, speed, label=""):
    """Move EE to target_xyz by applying delta from hover_T (Cartesian line)."""
    print(f"    → {label}  {[f'{v:.4f}' for v in xyz]}")
    # Build target SE3: copy hover_T rotation, update translation
    T = np.array(hover_T, dtype=np.float64)
    T[0, 3] = xyz[0]
    T[1, 3] = xyz[1]
    T[2, 3] = xyz[2]
    api.lin_go(T.tolist(), speed=speed)


def dip_slot(api, cal, slot_idx, ref_hover_T):
    """Move to a palette slot and dip. Uses joint move for ref_slot, Cartesian for others."""
    ref_slot   = int(cal["ref_slot"])
    hover_off  = float(cal["hover_z_offset"])
    slot_all   = cal.get("slot_xyz_all", {})
    r, g, b    = PALETTE_RGB[slot_idx]
    name       = PALETTE_NAMES[slot_idx]

    if slot_idx == ref_slot:
        # Use recorded joint angles
        q_h = np.array(cal["ref_hover_q"])
        q_d = np.array(cal["ref_dip_q"])
        _go_joint(api, q_h, HOVER_SPEED, f"hover slot {slot_idx} ({name})")
        _go_joint(api, q_d, DIP_SPEED,   f"dip into {name}")
        time.sleep(0.8)
        _go_joint(api, q_h, DIP_SPEED,   f"lift from {name}")
    else:
        # Compute Cartesian target from pre-computed positions
        if slot_idx in slot_all:
            hover_xyz = slot_all[slot_idx]["hover"]
            dip_xyz   = slot_all[slot_idx]["dip"]
        else:
            from palette_cfg import SLOT_GRID
            ref_row, ref_col = SLOT_GRID[ref_slot]
            si_row,  si_col  = SLOT_GRID[slot_idx]
            pitch_x, pitch_y = cal["slot_pitch_xy"]
            ref_dip  = cal["ref_dip_xyz"]
            dip_xyz  = [
                ref_dip[0] + (si_col - ref_col) * pitch_x,
                ref_dip[1] + (si_row - ref_row) * pitch_y,
                ref_dip[2],
            ]
            hover_xyz = list(dip_xyz)
            hover_xyz[2] += hover_off

        _go_cart(api, hover_xyz, ref_hover_T, cal["ref_hover_xyz"],
                 HOVER_SPEED, f"hover slot {slot_idx} ({name})")
        _go_cart(api, dip_xyz,   ref_hover_T, cal["ref_hover_xyz"],
                 DIP_SPEED,   f"dip into {name}")
        time.sleep(0.8)
        _go_cart(api, hover_xyz, ref_hover_T, cal["ref_hover_xyz"],
                 DIP_SPEED,   f"lift from {name}")


def wash(api, cal, n_rot, amp_deg):
    """Full wash cycle: hover → dip → cone sweep → hover."""
    q_hover = np.array(cal["water_hover_q"])
    q_dip   = np.array(cal["water_dip_q"])

    _go_joint(api, q_hover, HOVER_SPEED, "water cup hover")
    _go_joint(api, q_dip,   DIP_SPEED,   "dip into water")
    time.sleep(0.3)

    wps = cone_trajectory(q_dip, n_rot=n_rot, amp_deg=amp_deg)
    print(f"    → cone sweep  {n_rot} rot × {amp_deg}°  ({len(wps)} pts)")
    for wp in wps:
        api.joint_go(wp.tolist(), speed=CONE_SPEED)

    _go_joint(api, q_dip,   DIP_SPEED,   "re-centre")
    _go_joint(api, q_hover, DIP_SPEED,   "lift out")


def main():
    p = argparse.ArgumentParser(description="蘸色+涮笔 循环测试 (6 槽各一次)")
    p.add_argument("--ip",  required=True)
    p.add_argument("--cal", default=DEFAULT_CAL_PATH)
    p.add_argument("--n",   type=int,   default=2,   help="涮笔圈数 (默认 2)")
    p.add_argument("--amp", type=float, default=5.0, help="圆锥半角 度 (默认 5)")
    args = p.parse_args()

    cal_path = Path(args.cal)
    if not cal_path.exists():
        print(f"[ERROR] 未找到标定文件 {cal_path}  先运行 test2_calibrate.py")
        sys.exit(1)

    cal = np.load(str(cal_path), allow_pickle=True).item()
    ref_hover_T = np.array(cal["ref_hover_T"])

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
    print("  就绪。\n")

    print(f"  ══ 蘸色+涮笔循环  共 {N_SLOTS} 个颜色 ══\n")

    for i in range(N_SLOTS):
        r, g, b = PALETTE_RGB[i]
        print(f"\n  ─── 槽 {i}  {_swatch(r,g,b)} {PALETTE_NAMES[i]} ───")
        print(f"  [蘸墨]")
        dip_slot(api, cal, i, ref_hover_T)

        print(f"  [涮笔]")
        wash(api, cal, n_rot=args.n, amp_deg=args.amp)

    print("\n  ✓ 全部完成\n")


if __name__ == "__main__":
    main()
