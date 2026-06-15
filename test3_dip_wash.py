"""
test3_dip_wash.py — 轮流蘸墨 + 涮笔全流程测试（7种颜色各一次）

按标定文件，对每种颜色依次执行:
  goto_paint_hover → dip_paint → goto_water_hover
  → dip_water → cone_wash → lift_from_water → drip_wait

每段移动分三阶段 (升 → 平移 → 降)，避免斜向穿越颜料盘/水筒。
速度从 config.yaml [speeds] 读取，可用命令行参数覆盖。

Usage:
    python test3_dip_wash.py
    python test3_dip_wash.py --cal data/calibration/palette.npy --n 2 --amp 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from palette_cfg import SLOT_NAMES, SLOT_RGB, N_SLOTS, DEFAULT_CAL_PATH
from palette_actions import (
    goto_paint_hover, dip_paint,
    goto_water_hover, dip_water,
    cone_wash, lift_from_water, drip_wait,
    _speeds,
)
from wash_action    import CONE_N_ROT, CONE_AMP_DEG
from config_loader  import robot_ip


def _swatch(r, g, b):
    return f"\033[48;2;{r};{g};{b}m   \033[0m"


def main():
    spd = _speeds()   # read defaults from config

    p = argparse.ArgumentParser(description="蘸色+涮笔全流程测试")
    p.add_argument("--cal",   default=DEFAULT_CAL_PATH)
    p.add_argument("--n",     type=int,   default=CONE_N_ROT,        help="涮笔圈数")
    p.add_argument("--amp",   type=float, default=CONE_AMP_DEG,      help="圆锥半角 (度)")
    p.add_argument("--speed", type=float, default=spd["cone"],       help="涮笔角速度 (rad/s)")
    p.add_argument("--drip",  type=float, default=spd["drip_sec"],   help="滴水等待秒数")
    args = p.parse_args()

    cal_path = Path(args.cal)
    if not cal_path.exists():
        print(f"[ERROR] 未找到标定文件 {cal_path}  先运行 test2_calibrate.py")
        sys.exit(1)

    cal = np.load(str(cal_path), allow_pickle=True).item()

    ip = robot_ip()
    from franka import Franka
    robot = Franka(ip)
    if not robot.wait_ready():
        print("[ABORT] robot not ready")
        sys.exit(1)
    print()

    transit_z = float(np.array(cal["water_hover_xyz"])[2])
    print(f"  ══ 蘸色+涮笔循环  {N_SLOTS} 种颜色 ══")
    print(f"  transit_z = {transit_z:.4f} m  (Hover-2 高度)")
    print(f"  hover={spd['hover']}  dip={spd['dip']}  cone={args.speed} rad/s  drip={args.drip}s\n")

    for slot in range(N_SLOTS):
        r, g, b = SLOT_RGB[slot]
        print(f"\n  ── {_swatch(r,g,b)} {SLOT_NAMES[slot]} (slot {slot}) ──")

        goto_paint_hover(robot, cal, slot)
        dip_paint(robot, cal, slot)
        goto_water_hover(api, cal)
        dip_water(api, cal)
        cone_wash(robot, cal, n_rot=args.n, amp_deg=args.amp, speed=args.speed)
        lift_from_water(api, cal)
        drip_wait(args.drip)

    print("\n  ✓ 全部完成\n")


if __name__ == "__main__":
    main()
