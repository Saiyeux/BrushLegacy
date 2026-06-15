"""
test5_wash_dip.py — go home → 涮笔 → 蘸墨 → go home

Usage:
    python test5_wash_dip.py --color Red
    python test5_wash_dip.py --color 3          # slot index
    python test5_wash_dip.py --color Blue --cone_speed 0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from palette_cfg    import SLOT_NAMES, SLOT_RGB, N_SLOTS, DEFAULT_CAL_PATH
from palette_actions import (
    go_home,
    goto_water_hover, dip_water, cone_wash, lift_from_water, drip_wait,
    goto_paint_hover, dip_paint,
    _speeds,
)
from wash_action    import CONE_N_ROT, CONE_AMP_DEG
from config_loader  import load_config, robot_ip


def _resolve_slot(color_arg: str) -> int:
    if color_arg.isdigit():
        s = int(color_arg)
        if not 0 <= s < N_SLOTS:
            print(f"[ERROR] slot {s} 超出范围 (0–{N_SLOTS-1})")
            sys.exit(1)
        return s
    matches = [i for i, n in enumerate(SLOT_NAMES) if n.lower() == color_arg.lower()]
    if not matches:
        print(f"[ERROR] 未知颜色 '{color_arg}'")
        print(f"  可用: {', '.join(SLOT_NAMES)} 或 0–{N_SLOTS-1}")
        sys.exit(1)
    return matches[0]


def _swatch(r, g, b):
    return f"\033[48;2;{r};{g};{b}m   \033[0m"


def main():
    spd = _speeds()
    cfg = load_config()

    p = argparse.ArgumentParser(description="涮笔 + 蘸墨测试")
    p.add_argument("--color",      required=True,
                   help="颜色名称 (Red/Yellow/…) 或 slot 序号 (0–6)")
    p.add_argument("--cal",        default=DEFAULT_CAL_PATH)
    p.add_argument("--n",          type=int,   default=CONE_N_ROT,  help="涮笔圈数")
    p.add_argument("--amp",        type=float, default=CONE_AMP_DEG, help="圆锥半角 (度)")
    p.add_argument("--cone_speed", type=float, default=0.8,
                   help="涮笔角速度 rad/s (默认 0.8，稍快于移动)")
    args = p.parse_args()

    slot = _resolve_slot(args.color)
    r, g, b = SLOT_RGB[slot]

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

    print(f"  目标颜色: {_swatch(r,g,b)} {SLOT_NAMES[slot]} (slot {slot})")
    print(f"  移动速度: hover={spd['hover']}  dip={spd['dip']}")
    print(f"  涮笔速度: {args.cone_speed} rad/s × {args.n} 圈 × {args.amp}°\n")

    # ── 1. go home ────────────────────────────────────────────────────────────
    print("  [1/5] go home")
    go_home(robot)

    # ── 2. wash ───────────────────────────────────────────────────────────────
    print("\n  [2/5] 移动到水筒上方")
    goto_water_hover(robot, cal)

    print("\n  [3/5] 涮笔")
    dip_water(robot, cal)
    cone_wash(robot, cal, n_rot=args.n, amp_deg=args.amp, speed=args.cone_speed)
    lift_from_water(robot, cal)
    drip_wait()

    # ── 3. dip paint ──────────────────────────────────────────────────────────
    print(f"\n  [4/5] 蘸墨 → {SLOT_NAMES[slot]}")
    goto_paint_hover(robot, cal, slot)
    dip_paint(robot, cal, slot)

    # ── 4. go home ────────────────────────────────────────────────────────────
    print("\n  [5/5] go home")
    goto_water_hover(robot, cal)   # 先升到transit高度再回home，避免低位直接joint移动
    go_home(robot)

    print("\n  ✓ 完成\n")


if __name__ == "__main__":
    main()
