"""
test_calibrate_canvas.py — 标定画布平面（参照 Cobrush Pro calibrate.py）

两种模式，标定 TL 后选择：
  [g] 几何模式 — 输入 TL→TR / TL→BL 的 ΔX ΔY，其余角自动计算（推荐）
  [4] 4角模式  — 手动标定全部四角，SVD 平面拟合

输出: data/calibration/canvas.npy
  origin      : TL 在机器人坐标系的 XYZ
  xyz_rot     : 3×3，列为 [x_axis, y_axis, normal]
  width_m     : TL→TR 实测距离
  height_m    : TL→BL 实测距离
  z_canvas    : 画布平均 Z 高度
  corners_raw : 4×3，实测角点
  corners_proj: 4×3，拟合后角点
  width_px    : 画布图像分辨率（用于 pixel→robot 转换）
  height_px   : 同上

Usage:
    python test_calibrate_canvas.py
    python test_calibrate_canvas.py --out data/calibration/canvas.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from franka import Franka
from config_loader import robot_ip

CANVAS_CAL_PATH = "data/calibration/canvas.npy"


# ── helpers ───────────────────────────────────────────────────────────────────

def _record(robot: Franka, prompt: str):
    input(f"\n  → {prompt}\n    就位后按 Enter 记录 … ")
    st  = robot.read_state()
    T   = np.array(st.O_T_EE).reshape(4, 4, order='F')
    xyz = T[:3, 3].copy()
    q   = list(st.q)
    print(f"    已记录: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
    return xyz, q


def _read_float(prompt: str, default: float | None = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  请输入数字")


def _read_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("  请输入整数")


def _prompt_meta(cal_w_m: float, cal_h_m: float) -> dict:
    """Ask for display metadata that doesn't affect the physical calibration."""
    print(f"\n  实测尺寸: {cal_w_m*100:.1f} cm × {cal_h_m*100:.1f} cm")
    w_px = _read_int("  图像宽 width_px", 512)
    h_px = _read_int("  图像高 height_px", 512)
    margin = _read_float("  安全余量 canvas_margin_m (m)", 0.01)
    return {"width_px": w_px, "height_px": h_px, "canvas_margin_m": margin}


def _compute_workspace_limits(corners: np.ndarray) -> list:
    xy = corners[:, :2]
    return [float(xy[:, 0].min()), float(xy[:, 0].max()),
            float(xy[:, 1].min()), float(xy[:, 1].max())]


def _save(result: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), result)
    w, h = result["width_m"], result["height_m"]
    print(f"\n  ✓ 画布标定完成")
    print(f"  origin : {[round(v,4) for v in result['origin']]}")
    print(f"  尺寸   : {w*100:.2f} cm × {h*100:.2f} cm")
    print(f"  z      : {result['z_canvas']:.4f} m")
    print(f"  → {out_path}\n")


# ── geometry mode (recommended) ───────────────────────────────────────────────

def _geometry_mode(tl: np.ndarray, q_tl: list, out_path: Path):
    """TL + ΔX/ΔY vectors → compute all four corners without FK error."""
    z = float(tl[2])
    print("\n  几何模式：输入方向向量（机器人基坐标系，单位 m）")
    print("  提示：TL→TR 通常沿 -Y 方向，TL→BL 通常沿 -X 方向")

    dx_w = _read_float("TL→TR  ΔX (m)")
    dy_w = _read_float("TL→TR  ΔY (m)")
    dx_h = _read_float("TL→BL  ΔX (m)")
    dy_h = _read_float("TL→BL  ΔY (m)")

    tr = np.array([tl[0] + dx_w, tl[1] + dy_w, z])
    bl = np.array([tl[0] + dx_h, tl[1] + dy_h, z])
    br = np.array([tr[0] + dx_h, tr[1] + dy_h, z])

    x_axis = tr - tl;  width  = float(np.linalg.norm(x_axis));  x_axis /= width
    y_axis = bl - tl;  height = float(np.linalg.norm(y_axis));  y_axis /= height
    normal = np.cross(x_axis, y_axis);  normal /= np.linalg.norm(normal)
    xyz_rot = np.column_stack([x_axis, y_axis, normal])
    corners = np.array([tl, tr, br, bl])

    print(f"\n  各角点:")
    for lbl, c in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"    {lbl}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")

    meta = _prompt_meta(width, height)
    result = {
        "origin":           tl,
        "xyz_rot":          xyz_rot,
        "width_m":          width,
        "height_m":         height,
        "z_canvas":         z,
        "corners_raw":      corners,
        "corners_proj":     corners,
        "corner_joints":    np.array([q_tl, [0]*7, [0]*7, [0]*7]),
        "workspace_limits": _compute_workspace_limits(corners),
        **meta,
    }
    _save(result, out_path)


# ── 4-corner mode ─────────────────────────────────────────────────────────────

def _four_corner_mode(tl: np.ndarray, q_tl: list, robot: Franka, out_path: Path):
    """Record TR / BR / BL, SVD plane-fit, save."""
    corners_raw = [tl]
    qs          = [q_tl]
    for label in ["top-right (TR)", "bottom-right (BR)", "bottom-left (BL)"]:
        xyz, q = _record(robot, f"拖动到{label}")
        corners_raw.append(xyz)
        qs.append(q)

    tl_r, tr_r, br_r, bl_r = corners_raw
    pts      = np.array(corners_raw)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal   = Vt[-1]
    if normal[2] > 0:
        normal = -normal
    normal /= np.linalg.norm(normal)

    def proj(p):
        return p - np.dot(p - centroid, normal) * normal

    tl_p = proj(tl_r);  tr_p = proj(tr_r)
    bl_p = proj(bl_r);  br_p = tr_p + (bl_p - tl_p)

    x_axis = tr_p - tl_p;  width  = float(np.linalg.norm(x_axis));  x_axis /= width
    y_axis = bl_p - tl_p;  height = float(np.linalg.norm(y_axis));  y_axis /= height
    normal = np.cross(x_axis, y_axis);  normal /= np.linalg.norm(normal)
    xyz_rot  = np.column_stack([x_axis, y_axis, normal])
    z_canvas = float(np.mean([tl_p[2], tr_p[2], bl_p[2], br_p[2]]))

    print(f"\n  平面拟合 (raw → proj):")
    for lbl, raw, p in zip(["TL","TR","BL","BR"],
                            [tl_r,  tr_r,  bl_r,  br_r],
                            [tl_p,  tr_p,  bl_p,  br_p]):
        print(f"    {lbl}: z_raw={raw[2]:.4f}  z_proj={p[2]:.4f}  dz={p[2]-raw[2]:+.5f}")

    meta = _prompt_meta(width, height)
    result = {
        "origin":           tl_p,
        "xyz_rot":          xyz_rot,
        "width_m":          width,
        "height_m":         height,
        "z_canvas":         z_canvas,
        "corners_raw":      np.array(corners_raw),
        "corners_proj":     np.array([tl_p, tr_p, br_p, bl_p]),
        "corner_joints":    np.array(qs),
        "workspace_limits": _compute_workspace_limits(np.array(corners_raw)),
        **meta,
    }
    _save(result, out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="画布平面标定")
    p.add_argument("--out", default=CANVAS_CAL_PATH)
    args = p.parse_args()
    out_path = Path(args.out)

    ip    = robot_ip()
    robot = Franka(ip)
    if not robot.wait_ready():
        print("[ABORT] robot not ready")
        sys.exit(1)

    print("\n" + "="*55)
    print("  画布标定")
    print("  操作：按住引导按钮拖动笔尖到角点，松开后按 Enter")
    print("="*55)

    print("\n  步骤 1: 记录 TL（左上角）")
    tl, q_tl = _record(robot, "拖动笔尖到画布左上角 (TL)")

    print("\n  选择标定模式：")
    print("  [g] 几何模式 — 输入 TL→TR / TL→BL 向量（推荐，绕过 FK 误差）")
    print("  [4] 4角模式  — 继续手动标定 TR / BR / BL，SVD 平面拟合")
    while True:
        cmd = input("  > ").strip().lower()
        if cmd in ("g", ""):
            _geometry_mode(tl, q_tl, out_path)
            return
        elif cmd == "4":
            _four_corner_mode(tl, q_tl, robot, out_path)
            return
        else:
            print("  请输入 g 或 4")


if __name__ == "__main__":
    main()
