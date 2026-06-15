"""
test1_wash_dryrun.py — 空跑涮笔动作（不去任何标定位置）

从机械臂当前关节角直接执行 J5+J6 圆锥扫掠。
用于确认涮笔动作幅度、速度是否合适，无需任何标定文件。

Usage:
    python test1_wash_dryrun.py --ip 192.170.10.200
    python test1_wash_dryrun.py --ip 192.170.10.200 --n 3 --amp 5
"""

import argparse
import sys
import numpy as np

sys.path.insert(0, "src")
from wash_action import cone_trajectory, CONE_SPEED, DIP_SPEED, CONE_AMP_DEG, CONE_N_ROT


def main():
    p = argparse.ArgumentParser(description="空跑涮笔: J5+J6 圆锥扫掠，从当前位置出发")
    p.add_argument("--ip",  required=True, help="机械臂 IP")
    p.add_argument("--n",   type=int,   default=CONE_N_ROT,   help="旋转圈数 (默认 2)")
    p.add_argument("--amp", type=float, default=CONE_AMP_DEG, help="圆锥半角 度 (默认 5)")
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

    q_now = np.array(st.q)
    print(f"  当前关节角: {[f'{v:.3f}' for v in q_now]}")
    print(f"  圆锥参数: {args.n} 圈 × {args.amp}°\n")

    input("  ↑ 手动把笔放到水中合适位置，然后按 Enter 开始扫掠 … ")

    # 重新读取（位置可能被手动调整过）
    st = api.readOnce()
    q_now = np.array(st.q)

    waypoints = cone_trajectory(q_now, n_rot=args.n, amp_deg=args.amp)
    print(f"  开始扫掠 ({len(waypoints)} 个路点) …")

    for i, wp in enumerate(waypoints):
        api.joint_go(wp.tolist(), speed=CONE_SPEED)

    print("  ✓ 完成。调整 --n / --amp 后再次运行。\n")


if __name__ == "__main__":
    main()
