"""
test0_go_home.py — 连接机械臂 → 读取状态 → 回 home 位置

最基础的连通性测试。每次操作前后用来归位。

Usage:
    python test0_go_home.py
    python test0_go_home.py --speed 0.15   # 更慢
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")
from franka import Franka, HOME_JOINTS
from config_loader import load_config, robot_ip
from palette_actions import go_home


def _fmt_q(q):
    return "[" + ", ".join(f"{v:.4f}" for v in q) + "]"


def _fmt_T(T):
    import numpy as np
    arr = np.array(T).reshape(4, 4, order='F')
    xyz = arr[:3, 3]
    return f"xyz=[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]"


def main():
    cfg = load_config()
    p   = argparse.ArgumentParser(description="Go to home position")
    p.add_argument("--speed", type=float,
                   default=cfg.get("speeds", {}).get("hover", 0.2),
                   help="MotionGenerator speed (default from config)")
    args = p.parse_args()

    ip = robot_ip()
    robot = Franka(ip)
    if not robot.wait_ready():
        print("[ABORT] robot not ready")
        sys.exit(1)

    st = robot.read_state()
    print(f"  当前 q  : {_fmt_q(st.q)}")
    print(f"  当前 EE : {_fmt_T(st.O_T_EE)}")
    print(f"  目标 q  : {_fmt_q(HOME_JOINTS)}")
    print(f"  speed   : {args.speed}\n")

    go_home(robot, speed=args.speed)

    st2 = robot.read_state()
    print(f"\n  到达 q  : {_fmt_q(st2.q)}")
    print(f"  到达 EE : {_fmt_T(st2.O_T_EE)}")
    print("  ✓ done\n")


if __name__ == "__main__":
    main()
